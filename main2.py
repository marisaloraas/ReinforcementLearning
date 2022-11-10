import random
from contextlib import closing
from io import StringIO
from os import path
from typing import Optional
from keras.optimizer_v1 import Adam
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Embedding, Reshape
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory
import numpy as np
import gym
from gym import Env, spaces, utils
from gym.envs.toy_text.utils import categorical_sample

#new map
CITY = [
    "+-----------+",
    "| : : |=| :G|",
    "| :R: |=| : |",
    "| : : :+: : |",
    "| : | |=| : |",
    "| : | |=| : |",
    "| : : :+: :B|",
    "| | : |=| : |",
    "| |Y: |=| : |",
    "| : : :+: : |",
    "+-----------+"
]

#test map
CITY_TEST = [
    "+-----------+",
    "| : : | | :G|",
    "| :R: | | : |",
    "| : : : : : |",
    "| : | | | : |",
    "| : | | | : |",
    "| : : : : :B|",
    "| | : | | : |",
    "| |Y: | | : |",
    "| : : : : : |",
    "+-----------+"
]

#original map
MAP = [
    "+---------+",
    "|R: | : :G|",
    "| : | : : |",
    "| : : : : |",
    "| | : | : |",
    "|Y| : |B: |",
    "+---------+",
]
WINDOW_SIZE = (550*2, 350*2)


class TaxiEnv(Env):
    """
    The Taxi Problem
    from "Hierarchical Reinforcement Learning with the MAXQ Value Function Decomposition"
    by Tom Dietterich
    ### Description
    There are four designated locations in the grid world indicated by R(ed),
    G(reen), Y(ellow), and B(lue). When the episode starts, the taxi starts off
    at a random square and the passenger is at a random location. The taxi
    drives to the passenger's location, picks up the passenger, drives to the
    passenger's destination (another one of the four specified locations), and
    then drops off the passenger. Once the passenger is dropped off, the episode ends.
    Map:
        +---------+
        |R: | : :G|
        | : | : : |
        | : : : : |
        | | : | : |
        |Y| : |B: |
        +---------+
    ### Actions
    There are 6 discrete deterministic actions:
    - 0: move south
    - 1: move north
    - 2: move east
    - 3: move west
    - 4: pickup passenger
    - 5: drop off passenger
    ### Observations
    There are 500 discrete states since there are 25 taxi positions, 5 possible
    locations of the passenger (including the case when the passenger is in the
    taxi), and 4 destination locations.
    Note that there are 400 states that can actually be reached during an
    episode. The missing states correspond to situations in which the passenger
    is at the same location as their destination, as this typically signals the
    end of an episode. Four additional states can be observed right after a
    successful episodes, when both the passenger and the taxi are at the destination.
    This gives a total of 404 reachable discrete states.
    Passenger locations:
    - 0: R(ed)
    - 1: G(reen)
    - 2: Y(ellow)
    - 3: B(lue)
    - 4: in taxi
    Destinations:
    - 0: R(ed)
    - 1: G(reen)
    - 2: Y(ellow)
    - 3: B(lue)
    ### Rewards
    - -1 per step unless other reward is triggered.
    - +20 delivering passenger.
    - -10  executing "pickup" and "drop-off" actions illegally.
    state space is represented by:
    (taxi_row, taxi_col, passenger_location, destination)
    ### Arguments
    ```
    gym.make('Taxi-v3')
    ```
    ### Version History
    * v3: Map Correction + Cleaner Domain Description
    * v2: Disallow Taxi start location = goal location, Update Taxi observations in the rollout, Update Taxi reward threshold.
    * v1: Remove (3,2) from locs, add passidx<4 check
    * v0: Initial versions release
    """

    metadata = {"render_modes": ["human", "ansi", "rgb_array"], "render_fps": 4}

    def __init__(self):
        #self.desc = np.asarray(MAP, dtype="c")
        #self.locs = locs = [(0, 0), (0, 4), (4, 0), (4, 3)]
        self.desc = np.asarray(CITY, dtype="c")
        self.Q = [
            np.zeros((9,6,6), dtype=float), # R
            np.zeros((9,6,6), dtype=float), # G
            np.zeros((9,6,6), dtype=float), # Y
            np.zeros((9,6,6), dtype=float), # B
        ]
        #self.locs = locs = [(1, 1), (1, 7), (5, 0), (5, 5)]
        self.locs = locs = [(1, 1), (0, 5), (7, 1), (6, 5)]
        self.locs_colors = [(255, 0, 0), (0, 255, 0), (255, 255, 0), (0, 0, 255)]
        self.locs_stops = [(3, 2), (3, 5), (3, 8)]
        #num_states = 500
        num_states = 9 * 6 * 5 * 4
        #num_rows = 5
        num_rows = 9
        #num_columns = 5
        num_columns = 6
        max_row = num_rows - 1
        max_col = num_columns - 1
        self.initial_state_distrib = np.zeros(num_states)
        # self.stopped = True
        num_actions = 6
        self.P = {
            state: {action: [] for action in range(num_actions)}
            for state in range(num_states)
        }
        for row in range(num_rows):
            for col in range(num_columns):
                for pass_idx in range(len(locs) + 1):  # +1 for being inside taxi
                    for dest_idx in range(len(locs)):
                        state = self.encode(row, col, pass_idx, dest_idx)
                        if pass_idx < 4 and pass_idx != dest_idx:
                            self.initial_state_distrib[state] += 1
                        for action in range(num_actions):
                            # defaults
                            new_row, new_col, new_pass_idx = row, col, pass_idx
                            reward = (
                                -1
                            )  # default reward when there is no pickup/dropoff
                            done = False
                            taxi_loc = (row, col)
                            if action == 0:
                                if self.desc[row+1, col*2+1] == b"=" or self.desc[row+1, col*2+1] == b"+":
                                    reward = (-1)
                                    new_row = min(row+1, max_row)
                                elif self.desc[row+2, col*2+1] == b"-":
                                    # reward = (-5)
                                    new_row = min(row+1, max_row)
                                else:
                                    new_row = min(row+1, max_row)
                            elif action == 1:
                                if self.desc[row+1, col*2+1] == b"=" or self.desc[row+1, col*2+1] == b"+":
                                    #reward = (-2)
                                    new_row = max(row-1, 0)
                                elif self.desc[row, col*2+1] == b"-":
                                    # reward = (-5)
                                    new_row = max(row-1, 0)
                                else:
                                    new_row = max(row-1, 0)
                            elif action == 2:
                                if self.desc[row+1, 2*col+2] == b":":
                                    if self.desc[row+1, 2*col+1] == b"+":
                                        #reward = (-2)
                                        new_col = col+1
                                        # if self.stopped:
                                        #     self.stopped = False
                                        #     new_col = col
                                        # else:
                                        #     self.stopped = True
                                        #     new_col = col+1
                                    else:
                                        new_col = min(col+1, max_col)
                                else:
                                    # reward = (-5)
                                    new_col = col
                            elif action == 3:
                                if self.desc[row+1, 2*col] == b":":
                                    if self.desc[row+1, 2*col-1] == b"+":
                                        #reward = (-2)
                                        new_col = col-1
                                        # if self.stopped:
                                        #     self.stopped = False
                                        #     new_col = col
                                        # else:
                                        #     self.stopped = True
                                        #     new_col = col-1
                                    else:
                                        new_col = max(col-1, 0)
                                else:
                                    # reward = (-5)
                                    new_col = col
                            elif action == 4:
                                if pass_idx < 4 and taxi_loc == locs[pass_idx]:
                                    new_pass_idx = 4
                                else:
                                    reward = (-10)
                            elif action == 5:
                                if (taxi_loc == locs[dest_idx]) and pass_idx == 4:
                                    new_pass_idx = locs.index(taxi_loc)
                                    done = True
                                    reward = (20)
                                elif (taxi_loc in locs) and pass_idx == 4:
                                    new_pass_idx = locs.index(taxi_loc)
                                else:
                                    reward = -10
                            new_state = self.encode(
                                new_row, new_col, new_pass_idx, dest_idx
                            )
                            self.P[state][action].append((1.0, new_state, reward, done))
        self.initial_state_distrib /= self.initial_state_distrib.sum()
        self.action_space = spaces.Discrete(num_actions)
        self.observation_space = spaces.Discrete(num_states)

        # pygame utils
        self.window = None
        self.clock = None
        self.cell_size = (
            WINDOW_SIZE[0] / self.desc.shape[1],
            WINDOW_SIZE[1] / self.desc.shape[0],
        )
        self.taxi_imgs = None
        self.taxi_orientation = 0
        self.passenger_img = None
        self.destination_img = None
        self.median_horiz = None
        self.median_vert = None
        self.background_img = None
        self.stopsign_img = None
        self.highway_img = None

    def gen_episode(self, r=0, c=0, m=1000):
        states = [0]*(m+1)
        states[0] = (r,c)
        actions_chosen = [0]*(m+1)
        rewards = [0]*(m+1)
        returns = [0]*(m+1)

        (pass_idx, dest_idx) = random.sample(range(4), 2)

        curr = (r,c)

        for i in range(1, m+1):
            actions_chosen[i-1] = random.choices(range(6))[0]

            update = self.step(actions_chosen[i-1])
            curr = update[0]
            states[i] = update[0]
            rewards[i] = update[1]

        G = 0
        for i in range(m, -1, -1):
            returns[i] = G
            G = rewards[i] + 0.9 * G

        return (states, actions_chosen, rewards, returns)

    def encode(self, taxi_row, taxi_col, pass_loc, dest_idx):
        # (9) 6, 5, 4
        i = taxi_row
        i *= 6
        #i *= 5
        i += taxi_col
        i *= 5
        i += pass_loc
        i *= 4
        i += dest_idx
        return i

    def decode(self, i):
        out = []
        out.append(i % 4)
        i = i // 4
        out.append(i % 5)
        i = i // 5
        out.append(i % 6)
        i = i // 6
        #out.append(i % 5)
        #i = i // 5
        out.append(i)
        assert 0 <= i < 9
        #assert 0 <= i < 5
        return reversed(out)

    def step(self, a):
        transitions = self.P[self.s][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, d = transitions[i]
        self.s = s
        self.lastaction = a
        return (int(s), r, d, {"prob": p})

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self.s = categorical_sample(self.initial_state_distrib, self.np_random)
        self.lastaction = None
        self.taxi_orientation = 0
        if not return_info:
            return int(self.s)
        else:
            return int(self.s), {"prob": 1}

    def render(self, mode="human"):
        if mode == "ansi":
            return self._render_text()
        else:
            return self._render_gui(mode)

    def _render_gui(self, mode):
        import pygame  # dependency to pygame only if rendering with human

        if self.window is None:
            pygame.init()
            pygame.display.set_caption("Taxi")
            if mode == "human":
                self.window = pygame.display.set_mode(WINDOW_SIZE)
            else:  # "rgb_array"
                self.window = pygame.Surface(WINDOW_SIZE)
        if self.clock is None:
            self.clock = pygame.time.Clock()
        if self.taxi_imgs is None:
            file_names = [
                path.join(path.dirname(__file__), "img/cab_front.png"),
                path.join(path.dirname(__file__), "img/cab_rear.png"),
                path.join(path.dirname(__file__), "img/cab_right.png"),
                path.join(path.dirname(__file__), "img/cab_left.png"),
            ]
            self.taxi_imgs = [
                pygame.transform.scale(pygame.image.load(file_name), self.cell_size)
                for file_name in file_names
            ]
        if self.passenger_img is None:
            file_name = path.join(path.dirname(__file__), "img/passenger.png")
            self.passenger_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.destination_img is None:
            file_name = path.join(path.dirname(__file__), "img/hotel.png")
            self.destination_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
            self.destination_img.set_alpha(170)
        if self.median_horiz is None:
            file_names = [
                path.join(path.dirname(__file__), "img/gridworld_median_left.png"),
                path.join(path.dirname(__file__), "img/gridworld_median_horiz.png"),
                path.join(path.dirname(__file__), "img/gridworld_median_right.png"),
            ]
            self.median_horiz = [
                pygame.transform.scale(pygame.image.load(file_name), self.cell_size)
                for file_name in file_names
            ]
        if self.median_vert is None:
            file_names = [
                path.join(path.dirname(__file__), "img/gridworld_median_top.png"),
                path.join(path.dirname(__file__), "img/gridworld_median_vert.png"),
                path.join(path.dirname(__file__), "img/gridworld_median_bottom.png"),
            ]
            self.median_vert = [
                pygame.transform.scale(pygame.image.load(file_name), self.cell_size)
                for file_name in file_names
            ]
        if self.background_img is None:
            file_name = path.join(path.dirname(__file__), "img/taxi_background.png")
            self.background_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.stopsign_img is None:
            file_name = path.join(path.dirname(__file__), "img/stop.png")
            self.stopsign_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.highway_img is None:
            file_name = path.join(path.dirname(__file__), "img/highway.png")
            self.highway_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        desc = self.desc

        for y in range(0, desc.shape[0]):
            for x in range(0, desc.shape[1]):
                cell = (x * self.cell_size[0], y * self.cell_size[1])
                self.window.blit(self.background_img, cell)
                if desc[y][x] == b"+" and y != 0 and y != 10:
                    self.window.blit(self.stopsign_img, cell)
                if desc[y][x] == b"=" and y != 0 and y != 10:
                    self.window.blit(self.highway_img, cell)
                if desc[y][x] == b"|" and (y == 0 or desc[y - 1][x] != b"|"):
                    self.window.blit(self.median_vert[0], cell)
                elif desc[y][x] == b"|" and (
                    y == desc.shape[0] - 1 or desc[y + 1][x] != b"|"
                ):
                    self.window.blit(self.median_vert[2], cell)
                elif desc[y][x] == b"|":
                    self.window.blit(self.median_vert[1], cell)
                elif desc[y][x] == b"-" and (x == 0 or desc[y][x - 1] != b"-"):
                    self.window.blit(self.median_horiz[0], cell)
                elif desc[y][x] == b"-" and (
                    x == desc.shape[1] - 1 or desc[y][x + 1] != b"-"
                ):
                    self.window.blit(self.median_horiz[2], cell)
                elif desc[y][x] == b"-":
                    self.window.blit(self.median_horiz[1], cell)

        for cell, color in zip(self.locs, self.locs_colors):
            color_cell = pygame.Surface(self.cell_size)
            color_cell.set_alpha(128)
            color_cell.fill(color)
            loc = self.get_surf_loc(cell)
            self.window.blit(color_cell, (loc[0], loc[1] + 10))

        taxi_row, taxi_col, pass_idx, dest_idx = self.decode(self.s)

        if pass_idx < 4:
            self.window.blit(self.passenger_img, self.get_surf_loc(self.locs[pass_idx]))

        if self.lastaction in [0, 1, 2, 3]:
            self.taxi_orientation = self.lastaction
        dest_loc = self.get_surf_loc(self.locs[dest_idx])
        taxi_location = self.get_surf_loc((taxi_row, taxi_col))

        if dest_loc[1] <= taxi_location[1]:
            self.window.blit(
                self.destination_img,
                (dest_loc[0], dest_loc[1] - self.cell_size[1] // 2),
            )
            self.window.blit(self.taxi_imgs[self.taxi_orientation], taxi_location)
        else:  # change blit order for overlapping appearance
            self.window.blit(self.taxi_imgs[self.taxi_orientation], taxi_location)
            self.window.blit(
                self.destination_img,
                (dest_loc[0], dest_loc[1] - self.cell_size[1] // 2),
            )

        if mode == "human":
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window)), axes=(1, 0, 2)
            )

    def get_surf_loc(self, map_loc):
        return (map_loc[1] * 2 + 1) * self.cell_size[0], (
            map_loc[0] + 1
        ) * self.cell_size[1]

    def _render_text(self):
        desc = self.desc.copy().tolist()
        outfile = StringIO()

        out = [[c.decode("utf-8") for c in line] for line in desc]
        taxi_row, taxi_col, pass_idx, dest_idx = self.decode(self.s)

        def ul(x):
            return "_" if x == " " else x

        if pass_idx < 4:
            out[1 + taxi_row][2 * taxi_col + 1] = utils.colorize(
                out[1 + taxi_row][2 * taxi_col + 1], "yellow", highlight=True
            )
            pi, pj = self.locs[pass_idx]
            out[1 + pi][2 * pj + 1] = utils.colorize(
                out[1 + pi][2 * pj + 1], "blue", bold=True
            )
        else:  # passenger in taxi
            out[1 + taxi_row][2 * taxi_col + 1] = utils.colorize(
                ul(out[1 + taxi_row][2 * taxi_col + 1]), "green", highlight=True
            )

        di, dj = self.locs[dest_idx]
        out[1 + di][2 * dj + 1] = utils.colorize(out[1 + di][2 * dj + 1], "magenta")
        outfile.write("\n".join(["".join(row) for row in out]) + "\n")
        if self.lastaction is not None:
            outfile.write(
                f"  ({['South', 'North', 'East', 'West', 'Pickup', 'Dropoff'][self.lastaction]})\n"
            )
        else:
            outfile.write("\n")

        with closing(outfile):
            return outfile.getvalue()

    def close(self):
        if self.window is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()

def main2():
    env = gym.make("Taxi-v3")
    env.reset()
    action_size = (env.action_space.sample())
    #env.step(env.action_space.sample())[0]
    model = Sequential()
    model.add(Embedding(500, 10, input_length=1))
    model.add(Reshape((10,)))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(action_size, activation='linear'))
    print(model.summary())
    memory = SequentialMemory(limit=50000, window_length=1)
    policy = EpsGreedyQPolicy()
    dqn_only_embedding = DQNAgent(model=model, nb_actions=action_size, memory=memory, nb_steps_warmup=500,
                                  target_model_update=1e-2, policy=policy)
    dqn_only_embedding.compile(Adam(lr=1e-3), metrics=['mae'])
    dqn_only_embedding.fit(env, nb_steps=1000000, visualize=False, verbose=1, nb_max_episode_steps=99,
                           log_interval=100000)



def main():
    # Hyperparameters
    alpha = 0.1
    gamma = 0.6
    epsilon = 0.1
    env = TaxiEnv()
    observation, info = env.reset(seed=42, return_info=True)
    q_table = np.zeros([env.observation_space.n, env.action_space.n])
    # env.render()
    env.s = random.randint(0, 9 * 6 * 5 * 4)  # set environment to illustration's state
    # print(env.s)
    epochs = 0
    penalties, reward = 0, 0
    frames = []  # for animation
    done = False
    numsteps = 0
    # set num step in while loop
    # can put for loop around while loop here for several iterations of playing game
    for i in range(100000):
        print("training test " + str(i))
        state = env.reset()
        epochs, penalties, reward, = 0, 0, 0
        done = False
        while not done:
            if(random.uniform(0, 1) < epsilon):
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])
            observation, reward, done, info = env.step(action)
            old_value = q_table[state, action]
            next_max = np.max(q_table[observation])
            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            q_table[state, action] = new_value
            epochs += 1
            if reward == -10:
                penalties += 1

            # Put each rendered frame into dict for animation
            if reward == 20:
                done = True
            # if numsteps == 99:
            #     done = True
            state = observation
            epochs += 1
            #print("State :" + str(observation) + " reward: " + str(reward))
        # numsteps = 0
    # print_frames(frames)
    print("Training Data Finished\n")

    """Evaluate agent's performance after Q-learning"""

    total_epochs, total_penalties = 0, 0
    episodes = 100
    env.render()
    observation = env.reset()
    for _ in range(episodes):
        #print(q_table[observation])
        observation = env.reset()
        epochs, penalties, reward = 0, 0, 0
        done = False
        while not done:
            action = np.argmax(q_table[observation])
            if action == 4:
                print("QTable: ")
                print(q_table)
                print("\n  Q_table at passanger location:")
                print(q_table[observation])
            observation, reward, done, info = env.step(action)
            if reward == -10:
                penalties += 1
            if reward == 20:
                done = True
            frames.append({
                'frame': env.render(mode='human'),
                'state': observation,
                'action': action,
                'reward': reward
            }
            )
            epochs += 1

        total_penalties += penalties
        total_epochs += epochs

    print(f"Results after {episodes} episodes:")
    print(f"Average timesteps per episode: {total_epochs / episodes}")
    print(f"Average penalties per episode: {total_penalties / episodes}")
    env.close()

if __name__ == "__main__":
    main()
