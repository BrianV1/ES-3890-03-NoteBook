import gymnasium as gym
import numpy as np
import cv2
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
import os
import torch

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# =======================
# Custom Gym Environment
# =======================
class LunarTerrainEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super(LunarTerrainEnv, self).__init__()
        self.action_space = spaces.Discrete(4)  # Forward, Left, Right, Hover
        self.observation_space = spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)
        self.reset()

    def step(self, action):
        # Apply action effects
        if action == 0:  # Move Forward
            self.x += 0.2 * np.cos(self.angle)  # Step size
            self.y += 0.2 * np.sin(self.angle)
        elif action == 1:  # Turn Left
            self.angle -= 0.1
        elif action == 2:  # Turn Right
            self.angle += 0.1
        elif action == 3:  # Hover (No Movement)
            pass

        # Compute reward and termination flag
        reward, done = self._calculate_reward()

        # Get sensor readings (fake lidar)
        sensors = self._get_lidar_readings()

        obs = np.array([self.x, self.y, self.vx, self.vy] + sensors, dtype=np.float32)
        return obs, reward, done, False, {}

    def reset(self, seed=None, options=None):
        self.x, self.y = 0.0, 0.0  # Starting position
        self.vx, self.vy = 0.0, 0.0
        self.angle = 0.0
        self.goal = np.array([4.5, 4.5])  # Goal position
        self.obstacles = np.random.uniform(0.5, 4.5, size=(5, 2))
        self.prev_distance = np.linalg.norm(np.array([self.x, self.y]) - self.goal)  # Initialize distance
        obs = np.array([self.x, self.y, self.vx, self.vy] + self._get_lidar_readings(), dtype=np.float32)
        return obs, {}

    def _calculate_reward(self):
        current_distance = np.linalg.norm(np.array([self.x, self.y]) - self.goal)
        #print(f"distance: {current_distance}")
        delta = self.prev_distance - current_distance
        self.prev_distance = current_distance  # Update for next step

        # **1. Base reward: Encourage movement toward goal**
        reward = delta * 15  # Stronger reward for getting closer

        # **2. Mild time penalty** – Encourage faster completion
        reward -= 0.2  # Small negative reward per step

        # **3. Reduce the Path Efficiency Penalty** – Penalize major detours only
        direct_vector = self.goal - np.array([self.x, self.y])
        direct_angle = np.arctan2(direct_vector[1], direct_vector[0])  # Ideal direction
        angle_difference = abs((direct_angle - self.angle + np.pi) % (2 * np.pi) - np.pi)

        # Reduce impact of this penalty to avoid discouraging movement
        reward -= angle_difference * 1  

        # **4. Reduce Wall-Hugging Penalty** – Still discourage it but not excessively
        if self.x < 0.5 or self.x > 4.5 or self.y < 0.5 or self.y > 4.5:
            reward -= 1  # Weaken penalty to prevent excessive fear of walls

        # **5. Reduce the Collision Penalty** – Still harsh but not overwhelming
        if self._collided_with_wall() or self._collided_with_obstacle():
            print("Died")
            return reward -120, True  # Strong penalty, but not as extreme

        # **6. Near-Obstacle Penalty (Weaker)** – Discourage but don’t paralyze
        for obs in self.obstacles:
            distance_to_obstacle = np.linalg.norm(np.array([self.x, self.y]) - obs)
            if distance_to_obstacle < 0.5:  
                reward -= (0.5 - distance_to_obstacle) * 5  # Lower penalty

        # **7. Reduce Lidar-Based Penalty** – Warn the agent, don’t stop it
        lidar_readings = self._get_lidar_readings()
        min_lidar_distance = min(lidar_readings)
        if min_lidar_distance < 0.3:  
            reward -= (0.3 - min_lidar_distance) * 7  # Weakened penalty

        # **8. Reward for reaching the goal**
        if current_distance < 0.5:
            print("Found it!")
            return reward + 100, True  # Large reward for reaching goal

        return reward, False


    def _get_lidar_readings(self):
        distances = []
        for obs in self.obstacles:
            dist = np.linalg.norm(np.array([self.x, self.y]) - obs)
            distances.append(min(dist, 1.0))  # Cap at 1.0
        return distances[:2] + [1.0] * (2 - len(distances))  # Ensure exactly two lidar readings

    def _collided_with_wall(self):
        return self.x > 5 or self.x < 0 or self.y > 5 or self.y < 0  # Bounds check

    def _collided_with_obstacle(self):
        return any(np.linalg.norm(np.array([self.x, self.y]) - obs) < 0.3 for obs in self.obstacles)


# ====================================
# Train RL Model using Stable-Baselines3 on GPU
# ====================================
def train_model():
    # Create vectorized environment for training
    env = make_vec_env(LunarTerrainEnv, n_envs=4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model_path = "lunar_rl_model"

    # User choice: Update existing model or start fresh
    update_choice = input("Do you want to update the existing model if available? (y/n): ").strip().lower()
    if update_choice == "y" and os.path.exists(model_path + ".zip"):
        model = PPO.load(model_path, env=env, device=device)
        print("Resuming training from existing model.")
    else:
        model = PPO("MlpPolicy", env, device=device, verbose=1)
        print("Creating a new model from scratch.")

    total_timesteps = 100000
    for i in range(0, total_timesteps, 1000):
        model.learn(total_timesteps=1000)
        print(f"Trained {i+1000}/{total_timesteps} timesteps")

    model.save(model_path)
    print("Training Complete. Model Saved.")


# =========================
# OpenCV Visualization
# =========================
class LunarTerrainOpenCV:
    def __init__(self, env, size=500):
        self.env = env
        self.size = size
        self.scale = size / 5

    def render(self, obs):
        img = np.ones((self.size, self.size, 3), dtype=np.uint8) * 255

        # Draw goal
        goal_pos = (int(self.env.goal[0] * self.scale), int(self.env.goal[1] * self.scale))
        cv2.circle(img, goal_pos, 10, (0, 255, 0), -1)

        # Draw obstacles
        for o in self.env.obstacles:
            top_left = (int(o[0] * self.scale - 10), int(o[1] * self.scale - 10))
            bottom_right = (int(o[0] * self.scale + 10), int(o[1] * self.scale + 10))
            cv2.rectangle(img, top_left, bottom_right, (0, 0, 0), -1)

        # Draw agent
        agent_pos = (int(obs[0] * self.scale), int(obs[1] * self.scale))
        cv2.circle(img, agent_pos, 10, (0, 0, 255), -1)

        return img

    def run(self, model_path="lunar_rl_model"):
        model = PPO.load(model_path)
        obs, _ = self.env.reset()

        while True:
            action, _ = model.predict(obs)
            obs, reward, done, _, _ = self.env.step(action)

            # Debugging: Print agent's actions and distance to goal
            #print(f"Action: {action}, Reward: {reward}, Distance: {np.linalg.norm(np.array([self.env.x, self.env.y]) - self.env.goal)}")

            frame = self.render(obs)
            cv2.imshow("Lunar Navigation", frame)

            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            if done:
                obs, _ = self.env.reset()

        cv2.destroyAllWindows()


# =========================
# Main Script Entry Point
# =========================
if __name__ == "__main__":
    print("CUDA Device Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No CUDA Device")

    mode = input("Enter 'train' to train the model, or 'run' to visualize: ").strip().lower()

    if mode == "train":
        train_model()
    elif mode == "run":
        env = LunarTerrainEnv()
        visualizer = LunarTerrainOpenCV(env)
        visualizer.run()
    else:
        print("Invalid choice. Please enter 'train' or 'run'.")
