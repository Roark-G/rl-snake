import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Dict
import gymnasium as gym
from gymnasium import spaces
from enum import Enum
from IPython.display import display, clear_output

class Rewards(Enum):
    APPLE = 1.0
    DIE = 0
    STEP = 0
    APPLE_GAMMA = 0.99

class SnakeEnv(gym.Env):
    metadata = {"render_modes": ["ascii", "matplotlib"], "render_fps": 8}
    _DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # up, right, down, left

    def __init__(
        self,
        height: int = 10,
        width: int = 10,
        walls: bool = True,
        init_length: int = 3,
        seed: Optional[int] = None,
        render_mode: str = "ascii"
    ):
        super().__init__()

        assert render_mode in ["ascii", "matplotlib"], "render_mode must be 'ascii' or 'matplotlib'"
        self.render_mode = render_mode
        self.height = height
        self.width = width
        self.walls = walls
        
        assert init_length < width // 2, "init_length must not intersect with walls"
        self.init_length = init_length
        self._rng = np.random.default_rng(seed)

        self.action_space = spaces.Discrete(3)  # left, straight, right
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=((self.height * self.width * 2) + 4,),  # 1: apple, 1: snake + 4: directions
            dtype=np.float32,
        )

        # state
        self.board = np.zeros((self.height, self.width, 2), dtype=np.float32)
        self.observation_arr = np.zeros((self.height * self.width * (1 + 1 + 4),), dtype=np.float32)
        self.snake = None
        self.direction = None
        self.apple = None
        self.steps = 0
        self.steps_since_apple = 0
        self.max_steps_since_apple = height * width + 1 # leeway
        self.done = False

        # matplotlib state
        self.fig, self.ax, self.im = None, None, None

    # necessary for gym
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.steps = 0
        self.steps_since_apple = 0
        self.done = False

        center_r = self.height // 2
        start_c = self.width // 2 - (self.init_length // 2)
        self.snake = [(center_r, start_c + i) for i in range(self.init_length)][::-1]
        self.direction = 1  # right
        self.apple = self._sample_empty_cell()

        obs = self._get_obs()
        return obs, self._get_info()
    
    def _rotate_direction(self, turn: int):
        """Turn left (-1), straight (0), or right (+1)."""
        self.direction = (self.direction + turn) % 4

    def process_action(self, action: int) -> Tuple[Tuple[int, int], bool]:
        self._rotate_direction(action - 1)
        head_r, head_c = self.snake[0]
        dr, dc = self._DIRS[self.direction]
        new_head = (head_r + dr, head_c + dc)

        # wall collision
        if self.walls and not (0 <= new_head[0] < self.height and 0 <= new_head[1] < self.width):
            return new_head, True
        
        # wraparound
        if not self.walls:
            new_head = (new_head[0] % self.height, new_head[1] % self.width)

        # self collision
        if new_head in self.snake[:-1]:
            return new_head, True

        return new_head, False
    
    def calculate_apple_reward(self) -> np.float32:
        return Rewards.APPLE.value * (Rewards.APPLE_GAMMA.value ** self.steps_since_apple)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        if self.done:
            return self._get_obs(), 0.0, True, False, {}
        
        new_head, collision = self.process_action(action=action)
        self.steps += 1
        self.steps_since_apple += 1

        reward = Rewards.STEP.value
        terminated = False

        if collision:
            reward = Rewards.DIE.value
            terminated = True
            self.done = True
        else:
            ate = (new_head == self.apple)
            self.snake.insert(0, new_head)

            if ate:
                reward = self.calculate_apple_reward()
                self.steps_since_apple = 0
                self.apple = self._sample_empty_cell()
                if self.apple is None:
                    # Board full
                    terminated = True
                    self.done = True
            else:
                self.snake.pop()

            if self.steps_since_apple >= self.max_steps_since_apple:
                terminated = True
                self.done = True

        obs = self._get_obs()
        return obs, reward, terminated, False, self._get_info()

    def render(self):
        if self.render_mode == "ascii":
            self._render_ascii()
        elif self.render_mode == "matplotlib":
            self._render_matplotlib()

    def _render_ascii(self):
        grid = [[" ." for _ in range(self.width)] for _ in range(self.height)]
        for (r, c) in self.snake[1:]:
            grid[r][c] = " o"
        hr, hc = self.snake[0]
        grid[hr][hc] = " X"
        if self.apple:
            ar, ac = self.apple
            grid[ar][ac] = " @"
        print("\n".join("".join(row) for row in grid))
        print(f"Steps: {self.steps} | Length: {len(self.snake)}")

    def _render_matplotlib(self):
        # Extract original 2-channel board
        # channel 0 = apple
        # channel 1 = snake intensity
        apple = self.board[:, :, 0]
        snake = self.board[:, :, 1]

        # Build RGB image (H, W, 3)
        img = np.zeros((self.height, self.width, 3), dtype=np.float32)
        img[:, :, 0] = apple          # red channel
        img[:, :, 1] = snake          # green channel
        # blue stays 0

        if self.fig is None or self.ax is None:
            self.fig, self.ax = plt.subplots()
            self.im = self.ax.imshow(img, interpolation="nearest")
            self.ax.set_title("Snake")
            self.ax.axis("off")
        else:
            self.im.set_data(img)

        clear_output(wait=True)
        display(self.fig)

    def close(self):
        if self.fig:
            plt.close(self.fig)
        self.fig = self.ax = self.im = None

    def _update_board(self) -> np.ndarray:
        # Reset the 2-channel board:
        #   [:, :, 0] = apple
        #   [:, :, 1] = snake intensity
        self.board.fill(0)

        # Snake body
        for i, (r, c) in enumerate(self.snake):
            val = 1 - i / len(self.snake)    # 1 at head, decreasing toward tail
            self.board[r, c, 1] = val

        # Apple
        if self.apple:
            ar, ac = self.apple
            self.board[ar, ac, 0] = 1.0

        return self.board

    def _get_obs(self):
        board = self._update_board()              # (H, W, 2)
        board_flat = board.reshape(-1)            # (H*W*2,)

        direction = np.zeros(4, dtype=np.float32)
        direction[self.direction] = 1.0

        # print("Board shape:", board.shape, "Board flat shape:", board_flat.shape, "Direction shape:", direction.shape)
        return np.concatenate([board_flat, direction])

    def _get_info(self):
        return {"length": len(self.snake), "steps": self.steps}

    def _sample_empty_cell(self):
        occupied = set(self.snake)
        free = [(r, c) for r in range(self.height) for c in range(self.width) if (r, c) not in occupied]
        return None if not free else free[self._rng.integers(len(free))]