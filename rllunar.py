import gymnasium as gym
import numpy as np
import cv2
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import os
import torch
from scipy.ndimage import distance_transform_edt
from alive_progress import alive_bar

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# =======================
# Custom Gym Environment with Constant Speed Dynamics
# =======================
class LunarTerrainEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super(LunarTerrainEnv, self).__init__()
        # Observation: x, y in [0, 5], vx, vy in [-1, 1], five lidar readings in [0, 1]
        
        # Actions: 0 = Turn Left, 1 = Turn Right, 2 = Maintain Heading
        self.action_space = spaces.Discrete(3)
        self.constant_speed = 0.1
        self.goal_factor = 1.0
        self.safe_distance = 0.5
        self.obstacle_factor = 10.0
        self.large_penalty = 100.0
        self.large_bonus = 100.0

        self.lidar_samples = 360

        low_obs = np.array([0.0, 0.0, -1.0, -1.0] + [0.0]*self.lidar_samples , dtype=np.float32)
        high_obs = np.array([5.0, 5.0, 1.0, 1.0] + [7.1]*self.lidar_samples , dtype=np.float32)
        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)
        self.reset()

    def step(self, action):
        if action == 0:  # Turn Left
            self.angle -= 0.08
        elif action == 1:  # Turn Right
            self.angle += 0.08
        # For action == 2, maintain heading

        self.x += self.constant_speed * np.cos(self.angle)
        self.y += self.constant_speed * np.sin(self.angle)
        self.vx = self.constant_speed * np.cos(self.angle)
        self.vy = self.constant_speed * np.sin(self.angle)

        reward, done = self._calculate_reward()
        sensors = self._get_lidar_readings()
        obs = np.array([self.x, self.y, self.vx, self.vy], dtype=np.float32)
        obs = np.concatenate((obs, sensors), axis=0)
        return obs, reward, done, False, {}

    def reset(self, seed=None, options=None):
        self.x, self.y = 0.5, 0.5
        self.angle = np.arctan2(4.5 - self.y, 4.5 - self.x)
        self.vx = self.constant_speed * np.cos(self.angle)
        self.vy = self.constant_speed * np.sin(self.angle)
        self.goal = np.array([4.5, 4.5])
        self.obstacles = np.random.uniform(1.0, 4.0, size=(5, 2))

        # Compute distance transforms once per episode
        grid_size = 100
        conversion = grid_size / 5.0
        # Obstacle distance transform
        obstacle_grid = np.ones((grid_size, grid_size), dtype=np.uint8)
        for obs in self.obstacles:
            obs_pixel = (int(obs[0] * conversion), int(obs[1] * conversion))
            cv2.circle(obstacle_grid, obs_pixel, int(0.3 * conversion), 0, -1)
        self.obstacle_dt = distance_transform_edt(obstacle_grid) / conversion
        # Goal distance transform
        goal_grid = np.ones((grid_size, grid_size), dtype=np.uint8)
        goal_pixel = (int(self.goal[0] * conversion), int(self.goal[1] * conversion))
        cv2.circle(goal_grid, goal_pixel, int(0.3 * conversion), 0, -1)
        self.goal_dt = distance_transform_edt(goal_grid) / conversion

        # Initialize prev_goal_dt
        agent_pixel_x = min(max(int(self.x * conversion), 0), grid_size - 1)
        agent_pixel_y = min(max(int(self.y * conversion), 0), grid_size - 1)
        self.prev_goal_dt = self.goal_dt[agent_pixel_y, agent_pixel_x]

        sensors = self._get_lidar_readings()
        obs = np.array([self.x, self.y, self.vx, self.vy], dtype=np.float32)
        obs = np.concatenate((obs, sensors), axis=0)
        return obs, {}

    def _calculate_reward(self):
        current_distance = np.linalg.norm(np.array([self.x, self.y]) - self.goal)
        current_goal_dt = self._get_dt_value(self.goal_dt)
        obstacle_dt_value = self._get_dt_value(self.obstacle_dt)

        # Reward for progress toward the goal
        goal_delta = self.prev_goal_dt - current_goal_dt
        reward = goal_delta * self.goal_factor
        self.prev_goal_dt = current_goal_dt

        # Penalty for proximity to obstacles
        if obstacle_dt_value < self.safe_distance:
            obstacle_penalty = (self.safe_distance - obstacle_dt_value) * self.obstacle_factor
            reward -= obstacle_penalty

        # Check for collisions
        if self._collided_with_wall():
            print("❌ Collision with wall!")
            return reward - self.large_penalty, True
        if self._collided_with_obstacle():
            print("❌ Collision with obstacle!")
            return reward - self.large_penalty, True

        # Check for reaching the goal
        if current_distance < 0.3:
            print("✅ Goal reached!")
            return reward + self.large_bonus, True

        return reward, False

    def _get_dt_value(self, dt_grid):
        grid_size = dt_grid.shape[0]
        conversion = grid_size / 5.0
        agent_pixel_x = min(max(int(self.x * conversion), 0), grid_size - 1)
        agent_pixel_y = min(max(int(self.y * conversion), 0), grid_size - 1)
        return dt_grid[agent_pixel_y, agent_pixel_x]

    def _get_lidar_readings(self):
        max_range = 5.0  # maximum sensing distance (can be adjusted)
        # Precompute the beam angles (0 to 2π) for each of the lidar_samples beams.
        beam_angles = np.linspace(0, 2*np.pi, self.lidar_samples, endpoint=False)
        
        # Initialize an array to hold the closest obstacle distance for each beam.
        # We start with max_range for all beams.
        obstacle_distances = np.full(self.lidar_samples, max_range)
        
        # Get the robot's current position.
        robot_pos = np.array([self.x, self.y])
        
        # Loop over obstacles to update the distances where a beam hits an obstacle.
        for obs in self.obstacles:
            # Center of the obstacle.
            obs_center = obs
            # Vector from the robot to the obstacle.
            delta = obs_center - robot_pos
            d = np.linalg.norm(delta)
            if d == 0:
                continue  # avoid division by zero
            
            # Angle from the robot to the obstacle center.
            center_angle = np.arctan2(delta[1], delta[0])
            
            # Compute the half angular width of the obstacle as seen from the robot.
            # (If d < 0.3 then the robot is inside the obstacle; here we simply treat it as occluding all beams.)
            if d < 0.3:
                angle_offset = np.pi
            else:
                angle_offset = np.arcsin(min(0.3/d, 1.0))
            
            # Compute the angular difference between each beam and the obstacle center.
            # The np.angle trick gives differences in the range [-π, π].
            delta_angles = np.angle(np.exp(1j*(beam_angles - center_angle)))
            
            # Find the beams that are within the obstacle’s angular extent.
            mask = np.abs(delta_angles) <= angle_offset
            
            # For the beams in the mask, compute the intersection distance.
            # The projection of delta onto the beam direction gives t = d*cos(delta_angle).
            t_all = d * np.cos(delta_angles[mask])
            # The perpendicular distance is d*sin(delta_angle) (take absolute value).
            perp_all = np.abs(d * np.sin(delta_angles[mask]))
            
            # Only consider beams for which the obstacle is in front of the robot.
            valid = t_all > 0
            if not np.any(valid):
                continue
            
            t = t_all[valid]
            perp = perp_all[valid]
            # Compute the distance along the beam to the circle’s edge.
            # Clip any numerical issues with the square root by ensuring a nonnegative argument.
            sqrt_term = np.sqrt(np.maximum(0.0, 0.3**2 - perp**2))
            intersect_dist = t - sqrt_term
            
            # Map these valid beams back to their indices in the full array.
            beam_indices = np.nonzero(mask)[0][valid]
            # Update the distance for these beams if the computed intersection is closer.
            obstacle_distances[beam_indices] = np.minimum(obstacle_distances[beam_indices], intersect_dist)
        
        # Now compute the distances to the grid walls for each beam.
        # The grid has walls at x = 0, x = 5, y = 0, and y = 5.
        dx = np.cos(beam_angles)
        dy = np.sin(beam_angles)
        
        # Distance to vertical walls.
        dist_x = np.where(dx > 0, (5.0 - self.x) / dx,
                        np.where(dx < 0, self.x / -dx, np.inf))
        # Distance to horizontal walls.
        dist_y = np.where(dy > 0, (5.0 - self.y) / dy,
                        np.where(dy < 0, self.y / -dy, np.inf))
        wall_distances = np.minimum(dist_x, dist_y)
        
        # The final lidar reading is the minimum of the obstacle distance and the wall distance.
        readings = np.minimum(obstacle_distances, wall_distances)
        
        return readings

    def _collided_with_wall(self):
        return self.x < 0 or self.x > 5 or self.y < 0 or self.y > 5

    def _collided_with_obstacle(self):
        return any(np.linalg.norm(np.array([self.x, self.y]) - obs) < 0.3 for obs in self.obstacles)

# ====================================
# Train RL Model using Stable-Baselines3
# ====================================
def train_model():
    env = make_vec_env(LunarTerrainEnv, n_envs=4)
    device = "cpu" 
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    print(f"Using device: {device}")

    model_path = "lunar_rl_model"
    update_choice = input("Do you want to update the existing model if available? (y/n): ").strip().lower()
    if update_choice == "y" and os.path.exists(model_path + ".zip"):
        model = PPO.load(model_path, env=env, device=device)
        print("Resuming training from existing model.")
    else:
        model = PPO("MlpPolicy", env, learning_rate=1e-4, device=device, verbose=1)
        print("Creating a new model from scratch.")

    total_timesteps = 50000
    with alive_bar(total_timesteps, title="Training Progress") as bar:
        for i in range(0, total_timesteps, 1000):
            model.learn(total_timesteps=1000)
            print(f"Trained {i+1000}/{total_timesteps} timesteps")
            bar(1000)
    model.save(model_path)
    print("Training Complete. Model Saved.")

# =========================
# OpenCV Visualization with Distance Transform Rendering
# =========================
class LunarTerrainOpenCV:
    def __init__(self, env, size=500):
        self.env = env
        self.size = size
        self.scale = size / 5

    def render(self, obs):
        img = np.ones((self.size, self.size, 3), dtype=np.uint8) * 255
        goal_pos = (int(self.env.goal[0] * self.scale), int(self.env.goal[1] * self.scale))
        cv2.circle(img, goal_pos, 10, (0, 255, 0), -1)
        for o in self.env.obstacles:
            top_left = (int(o[0] * self.scale - 10), int(o[1] * self.scale - 10))
            bottom_right = (int(o[0] * self.scale + 10), int(o[1] * self.scale + 10))
            cv2.rectangle(img, top_left, bottom_right, (0, 0, 0), -1)
        agent_pos = (int(obs[0] * self.scale), int(obs[1] * self.scale))
        cv2.circle(img, agent_pos, 10, (0, 0, 255), -1)
        return img

    def render_distance_transform(self):
        grid_size = 100
        grid = np.ones((grid_size, grid_size), dtype=np.uint8)
        conversion = grid_size / 5.0
        obstacle_radius = int(0.3 * conversion)
        for o in self.env.obstacles:
            obs_pixel = (int(o[0] * conversion), int(o[1] * conversion))
            cv2.circle(grid, obs_pixel, obstacle_radius, 0, -1)
        dt = distance_transform_edt(grid)
        dt_norm = cv2.normalize(dt, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        dt_color = cv2.applyColorMap(dt_norm, cv2.COLORMAP_JET)
        agent_pixel = (int(self.env.x * conversion), int(self.env.y * conversion))
        cv2.circle(dt_color, agent_pixel, 3, (0, 0, 0), -1)
        dt_color = cv2.resize(dt_color, (self.size, self.size))
        return dt_color

    def run(self, model_path="lunar_rl_model"):
        model = PPO.load(model_path)
        obs, _ = self.env.reset()

        while True:
            action, _ = model.predict(obs)
            obs, reward, done, _, _ = self.env.step(action)
            dist = np.linalg.norm(np.array([self.env.x, self.env.y]) - self.env.goal)
            print(f"Distance to goal: {dist:.2f}, Reward: {reward:.2f}")
            frame = self.render(obs)
            dt_frame = self.render_distance_transform()
            cv2.imshow("Lunar Navigation", frame)
            cv2.imshow("Distance Transform", dt_frame)
            key = cv2.waitKey(100) & 0xFF
            if key == ord('q'):
                break
            if done:
                obs, _ = self.env.reset()
        cv2.destroyAllWindows()

# =========================
# Main Script Entry Point
# =========================
if __name__ == "__main__":
    print("CUDA Device Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No CUDA Device")
    mode = input("Enter 'train' to train the model, 'run' to visualize, or 'dt' to view distance transform: ").strip().lower()
    if mode == "train":
        train_model()
    elif mode == "run":
        env = LunarTerrainEnv()
        visualizer = LunarTerrainOpenCV(env)
        visualizer.run()
    elif mode == "dt":
        env = LunarTerrainEnv()
        _, _ = env.reset()
        visualizer = LunarTerrainOpenCV(env)
        dt_frame = visualizer.render_distance_transform()
        cv2.imshow("Distance Transform", dt_frame)
        print("Press any key to exit the distance transform view.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Invalid choice. Please enter 'train', 'run', or 'dt'.")