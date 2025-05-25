import numpy as np
import random


class LunarLanderEnvironment:
    def __init__(self, screen_width=800, screen_height=600):
        # Physics constants - ADJUSTED FOR BETTER LEARNING
        self.GRAVITY = 1.22  # Reduced moon's gravity for easier control (was 1.62)
        self.MAX_VELOCITY = 7.0  # Increased safe landing velocity (was 5.0)
        self.INITIAL_FUEL = 1200  # More starting fuel for longer episodes (was 1000)
        self.FUEL_CONSUMPTION_RATE = 1  # Unchanged

        # Screen and visualization setup
        self.screen_width = screen_width
        self.screen_height = screen_height

        # Landing zone properties - WIDER LANDING ZONE
        self.landing_zone_width = 150  # Wider landing zone (was 100)
        self.landing_zone_x = self.screen_width / 2

        # Performance tracking
        self.episode_rewards = []
        self.success_rate = []
        self.trajectory = []  # Store positions for trajectory trail

        # Keep track of landing outcomes
        self.landing_successful = False
        self.landing_speed = 0
        self.landing_position = 0

        # Lander initial state
        self.reset()

    def reset(self):
        """Reset the lander to initial state."""
        # Starting position with reduced randomization
        self.x = self.screen_width / 2 + random.uniform(-30, 30)  # Less initial horizontal offset (was -50, 50)
        self.y = 120  # Slightly higher starting point for more time to learn (was 100)

        # Initial velocities with less variability
        self.vertical_velocity = random.uniform(-0.5, 0.5)  # Added small initial vertical velocity
        self.horizontal_velocity = random.uniform(-0.8, 0.8)  # Reduced initial horizontal velocity (was -1, 1)

        # Fuel and state tracking
        self.fuel = self.INITIAL_FUEL
        self.time_steps = 0

        # Reset trajectory
        self.trajectory = [(self.x, self.y)]

        # Reset thrust visualization flags
        self.thrusting_main = False
        self.thrusting_left = False
        self.thrusting_right = False

        # Reset landing status
        self.landing_successful = False
        self.landing_speed = 0
        self.landing_position = 0

        return self._get_state()

    def _get_state(self):
        """
        Returns the current state of the lander.
        """
        landing_x = self.landing_zone_x
        landing_zone_width = self.landing_zone_width

        # Enhanced state with normalized values
        return np.array([
            self.y / self.screen_height,  # Vertical position
            self.vertical_velocity / self.MAX_VELOCITY,  # Vertical velocity
            self.horizontal_velocity / self.MAX_VELOCITY,  # Horizontal velocity
            self.fuel / self.INITIAL_FUEL,  # Remaining fuel
            (self.x - landing_x) / (self.screen_width / 2),  # Horizontal distance to landing zone
            self.x / self.screen_width  # Normalized horizontal position
        ])

    def step(self, action):
        """
        Execute a single step in the environment.
        Actions: 0 (no thrust), 1 (left thrust), 2 (right thrust), 3 (main thrust)
        """
        # Reset thrusting flags
        self.thrusting_main = False
        self.thrusting_left = False
        self.thrusting_right = False

        # Apply gravity
        self.vertical_velocity += self.GRAVITY * 0.1

        # CRITICAL FIX: Clamp vertical velocity to prevent excessive upward movement
        self.vertical_velocity = min(12.0, self.vertical_velocity)  # Prevent excessive downward speed

        # Apply thrust based on action with improved control
        thrust_power = 5.0  # Increased from 4.0 for more responsive controls

        if action == 1 and self.fuel > 0:  # Left thrust
            self.horizontal_velocity += thrust_power * 0.1
            self.fuel -= self.FUEL_CONSUMPTION_RATE
            self.thrusting_left = True
        elif action == 2 and self.fuel > 0:  # Right thrust
            self.horizontal_velocity -= thrust_power * 0.1
            self.fuel -= self.FUEL_CONSUMPTION_RATE
            self.thrusting_right = True
        elif action == 3 and self.fuel > 0:  # Main thrust
            # CRITICAL FIX: Add non-linearity to main thrust - more powerful when moving fast
            thrust_multiplier = 0.2 + 0.1 * min(1.0, self.vertical_velocity / 5.0)
            thrust_amount = thrust_power * thrust_multiplier

            # Ensure we don't get excessive upward velocity
            if self.vertical_velocity > -8.0:  # Only apply full thrust if not moving up too fast
                self.vertical_velocity -= thrust_amount
                # Add slight damping to prevent bouncing effect
                if self.vertical_velocity < 0:
                    self.vertical_velocity *= 0.98
            self.fuel -= self.FUEL_CONSUMPTION_RATE
            self.thrusting_main = True

        # Update position with velocity damping (slight air resistance)
        self.horizontal_velocity *= 0.995  # Small horizontal damping
        self.x += self.horizontal_velocity
        self.y += self.vertical_velocity

        # Store position for trajectory
        self.trajectory.append((self.x, self.y))
        # Limit trajectory length to prevent performance issues
        if len(self.trajectory) > 50:
            self.trajectory.pop(0)

        # Calculate reward
        reward = self._calculate_reward(action)

        # Check termination conditions
        done, terminal_reward = self._check_termination()

        # Add terminal reward to step reward
        if done:
            reward += terminal_reward

        self.time_steps += 1

        return self._get_state(), reward, done

    def _calculate_reward(self, action):
        """Enhanced reward shaping for better learning"""
        # Starting with very small negative reward to encourage efficiency
        reward = -0.01  # Small penalty for each step

        # Landing zone parameters
        landing_x = self.landing_zone_x
        landing_zone_width = self.landing_zone_width

        # Distance to landing zone center - to encourage moving toward landing zone
        distance_to_center = abs(self.x - landing_x)
        normalized_distance = min(1.0, distance_to_center / (self.screen_width / 2))

        # Progressive reward for getting closer to landing zone - INCREASED
        reward += 0.08 * (1.0 - normalized_distance)  # Was 0.05, increased to emphasize position

        # IMPROVED: Height-based reward to encourage staying in control
        height_ratio = self.y / self.screen_height
        ground_proximity = 1.0 - height_ratio  # 0 at top, 1 near ground

        # Vertical velocity rewards - COMPLETELY RESTRUCTURED
        # Encourage counterthrust when falling too fast
        if self.vertical_velocity > self.MAX_VELOCITY * 0.7:
            # Stronger penalty when close to the ground and falling fast
            velocity_penalty = min(0.3, 0.05 * (self.vertical_velocity / self.MAX_VELOCITY))
            proximity_factor = ground_proximity ** 2  # Exponential importance near ground
            reward -= velocity_penalty * proximity_factor * 2.0

        # Reward for maintaining safe vertical velocity near ground
        if self.y > self.screen_height * 0.7:  # Near ground
            if abs(self.vertical_velocity) < self.MAX_VELOCITY * 0.8:
                reward += 0.05 * ground_proximity  # More reward closer to ground

        # Horizontal control rewards - IMPROVED
        # Stronger reward for being centered over landing zone
        if abs(self.x - landing_x) < landing_zone_width * 0.3:  # Near center of landing zone
            reward += 0.1  # Significant reward for being well-centered

        # Reward for maintaining low horizontal velocity - INCREASED
        horizontal_control = max(0, 1.0 - abs(self.horizontal_velocity) / self.MAX_VELOCITY)
        reward += 0.05 * horizontal_control  # More reward for slower horizontal speed

        # ENGINE USAGE REWARDS - IMPROVED
        # Reward for using main engine appropriately when falling fast
        if action == 3 and self.vertical_velocity > 0:  # Using main engine while falling
            # More reward for using engine when falling faster
            engine_reward = min(0.1, 0.02 * (self.vertical_velocity / self.MAX_VELOCITY))
            reward += engine_reward

        # Small reward for moving toward landing zone
        if (landing_x > self.x and action == 1) or (landing_x < self.x and action == 2):
            reward += 0.02

        # NEW: Stability reward - encourage staying level while descending
        if abs(self.horizontal_velocity) < 2.0 and 0 < self.vertical_velocity < 3.0:
            reward += 0.05  # Reward stable descent

        # NEW: Extra reward for hovering above landing zone
        if landing_x - landing_zone_width / 2 < self.x < landing_x + landing_zone_width / 2:
            if self.y > self.screen_height * 0.7:  # Near ground
                if abs(self.vertical_velocity) < 2.0:  # Hovering
                    reward += 0.1  # Significant reward for controlled hovering above landing zone

        return reward

    def _check_termination(self):
        """Check if episode should terminate and calculate terminal rewards"""
        done = False
        terminal_reward = 0
        landing_zone_width = self.landing_zone_width
        landing_x = self.landing_zone_x

        # Out of bounds - terminate with large penalty
        # IMPROVED: More forgiving out-of-bounds conditions
        if (self.x < -50 or self.x > self.screen_width + 50 or  # Wider horizontal bounds
                self.y > self.screen_height or  # Still terminate if hitting ground
                self.y < -50):  # More forgiving upper bound

            done = True
            # REDUCED PENALTY for out-of-bounds to prevent excessive negative bias
            terminal_reward = -20  # Was -30
            self.landing_successful = False
            return done, terminal_reward

        # Landing conditions - detect when lander is at ground level
        ground_level = self.screen_height - 50
        if self.y >= ground_level:
            done = True

            # Record landing metrics
            self.landing_speed = abs(self.vertical_velocity)
            self.landing_position = self.x

            # Check if landed in landing zone (more lenient condition)
            in_landing_zone = (landing_x - landing_zone_width / 2 < self.x < landing_x + landing_zone_width / 2)

            if in_landing_zone:
                # IMPROVED SUCCESS CONDITIONS: Much more lenient velocity thresholds
                if (abs(self.vertical_velocity) <= self.MAX_VELOCITY * 1.4 and  # 40% more lenient (was 1.2)
                        abs(self.horizontal_velocity) <= self.MAX_VELOCITY * 1.4):  # 40% more lenient (was 1.2)

                    # Calculate landing quality factors
                    speed_factor = 1.0 - min(1.0, (abs(self.vertical_velocity) / (self.MAX_VELOCITY * 1.4)))
                    position_factor = 1.0 - min(1.0, (abs(self.x - landing_x) / (landing_zone_width / 2)))
                    fuel_factor = self.fuel / self.INITIAL_FUEL

                    # INCREASED success reward
                    terminal_reward = 250 + 70 * speed_factor + 30 * position_factor + 25 * fuel_factor  # Was 200
                    self.landing_successful = True
                    print(f"SUCCESS! Speed: {self.landing_speed:.2f}, Position: {abs(self.x - landing_x):.2f}")
                else:
                    # Hard landing - crashed but in landing zone
                    # SOFTER penalty for crash in zone
                    proximity_to_success = 1.0 - min(1.0,
                                                     (
                                                                 abs(self.vertical_velocity) - self.MAX_VELOCITY * 1.4) / self.MAX_VELOCITY)
                    terminal_reward = -10 + 5 * proximity_to_success  # Less severe penalty (was -15)
                    self.landing_successful = False
                    print(f"CRASH IN ZONE: VS={self.vertical_velocity:.2f}, HS={self.horizontal_velocity:.2f}")
            else:
                # Crashed outside landing zone
                dist_to_landing_zone = min(
                    abs(self.x - (landing_x - landing_zone_width / 2)),
                    abs(self.x - (landing_x + landing_zone_width / 2))
                )
                proximity_factor = 1.0 - min(1.0, dist_to_landing_zone / (self.screen_width / 4))
                # SMALLER PENALTY for missing zone
                terminal_reward = -20 + 15 * proximity_factor  # Reduced penalty (was -25)
                self.landing_successful = False
                print(f"MISSED ZONE: Missed by {dist_to_landing_zone:.2f} pixels")

        return done, terminal_reward