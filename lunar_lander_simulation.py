import pygame
import numpy as np
import torch
import math
import os
import time
import random
import argparse
import matplotlib.pyplot as plt
from lunar_lander_env import LunarLanderEnvironment
from deep_q_agent import DQNAgent


class LunarLanderSimulation:
    def __init__(self, mode="train", visualize=True):
        # Initialize visualization flag
        self.visualize = visualize

        # Initialize Pygame only if visualization is enabled
        if self.visualize:
            pygame.init()
            # Screen setup
            self.screen_width = 800
            self.screen_height = 600
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Lunar Lander Simulation")

            # Enhanced colors and visuals
            self.BACKGROUND = (5, 5, 20)  # Dark blue background
            self.WHITE = (255, 255, 255)
            self.GRAY = (150, 150, 150)
            self.BLACK = (0, 0, 0)
            self.RED = (255, 50, 50)
            self.GREEN = (50, 255, 50)
            self.BLUE = (50, 50, 255)
            self.YELLOW = (255, 255, 0)
            self.ORANGE = (255, 165, 0)

            # Particle effects
            self.particles = []
            self.stars = self._generate_stars(150)  # Generate 150 stars for background

            # Clock for controlling frame rate
            self.clock = pygame.time.Clock()

            # Font for displaying stats
            self.font = pygame.font.Font(None, 24)
            self.title_font = pygame.font.Font(None, 36)
        else:
            # Set these properties even when not visualizing
            self.screen_width = 800
            self.screen_height = 600
    def __init__(self):
        # Initialize Pygame
        pygame.init()

        # Screen setup
        self.screen_width = 800
        self.screen_height = 600
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Lunar Lander Simulation")

        # Enhanced colors and visuals
        self.BACKGROUND = (5, 5, 20)  # Dark blue background
        self.WHITE = (255, 255, 255)
        self.GRAY = (150, 150, 150)
        self.BLACK = (0, 0, 0)
        self.RED = (255, 50, 50)
        self.GREEN = (50, 255, 50)
        self.BLUE = (50, 50, 255)
        self.YELLOW = (255, 255, 0)
        self.ORANGE = (255, 165, 0)

        # Particle effects
        self.particles = []
        self.stars = self._generate_stars(150)  # Generate 150 stars for background

        # Create directories for saving
        os.makedirs("models", exist_ok=True)
        os.makedirs("stats", exist_ok=True)

        # Environment and Agent
        self.env = LunarLanderEnvironment(self.screen_width, self.screen_height)
        self.agent = DQNAgent(state_dim=6, action_dim=4, learning_rate=0.0005)

        # Training parameters - adjusted for better learning
        self.num_episodes = 2000  # More episodes for stable learning
        self.max_steps_per_episode = 800  # Longer episodes to help learn landing
        self.target_update_frequency = 4  # Update target network frequently
        # Clock for controlling frame rate
        self.clock = pygame.time.Clock()

        # Font for displaying stats
        self.font = pygame.font.Font(None, 24)  # Smaller font for more stats
        self.title_font = pygame.font.Font(None, 36)  # Larger font for titles

        # Environment and Agent
        self.env = LunarLanderEnvironment(self.screen_width, self.screen_height)
        self.agent = DQNAgent(state_dim=6, action_dim=4, learning_rate=0.0005)  # Updated state dimension

        # Training parameters
        self.num_episodes = 500
        self.max_steps_per_episode = 500
        self.target_update_frequency = 5  # Update target network more frequently

        # Stats tracking
        self.episode_rewards = []
        self.avg_rewards = []
        self.best_reward = -float('inf')
        self.successful_landings = 0
        self.last_10_success_rate = 0
        self.training_start_time = time.time()

        # Progress tracking
        self.success_window = []  # Track success in rolling window
        self.reward_window = []  # Track rewards in rolling window

        # Mode (train or play)
        self.mode = mode

    def _generate_stars(self, num_stars):
        """Generate random stars for the background"""
        if not self.visualize:
            return []

        self.training_start_time = time.time()

    def _generate_stars(self, num_stars):
        """Generate random stars for the background"""
        stars = []
        for _ in range(num_stars):
            x = random.randint(0, self.screen_width)
            y = random.randint(0, self.screen_height)
            size = random.randint(1, 3)
            brightness = random.randint(150, 255)
            stars.append((x, y, size, brightness))
        return stars

    def _draw_stars(self):
        """Draw stars in the background"""
        if not self.visualize:
            return

        for x, y, size, brightness in self.stars:
            # Make stars twinkle slightly
            flicker = random.randint(-20, 20)
            b = max(100, min(255, brightness + flicker))
            pygame.draw.circle(self.screen, (b, b, b), (x, y), size)

    def _create_particles(self, x, y, color, count=5):
        """Create engine exhaust particles"""
        if not self.visualize:
            return

        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 3)
            size = random.randint(1, 3)
            lifetime = random.randint(5, 15)
            velocity_x = math.cos(angle) * speed
            velocity_y = math.sin(angle) * speed
            self.particles.append({
                'x': x, 'y': y,
                'vx': velocity_x, 'vy': velocity_y,
                'size': size, 'color': color,
                'lifetime': lifetime
            })

    def _update_particles(self):
        """Update and remove particles"""
        if not self.visualize:
            return

        active_particles = []
        for p in self.particles:
            # Update position
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['lifetime'] -= 1

            # Keep particle if still alive
            if p['lifetime'] > 0:
                active_particles.append(p)

        self.particles = active_particles

    def _draw_particles(self):
        """Draw all active particles"""
        if not self.visualize:
            return

        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], (int(p['x']), int(p['y'])), p['size'])

    def draw_lander(self, x, y):
        if not self.visualize:
            return

        # Enhanced lander design
        # Base body
        lander_points = [
            (x, y),  # Top center
            (x - 15, y + 15),  # Top left
            (x - 15, y + 40),  # Bottom left
            (x + 15, y + 40),  # Bottom right
            (x + 15, y + 15),  # Top right
        ]
        pygame.draw.polygon(self.screen, self.WHITE, lander_points)

        # Landing legs
        pygame.draw.line(self.screen, self.GRAY, (x - 15, y + 40), (x - 25, y + 50), 3)
        pygame.draw.line(self.screen, self.GRAY, (x + 15, y + 40), (x + 25, y + 50), 3)

        # Thruster visualizations
        if self.env.thrusting_main:
            # Main thruster flame
            flame_points = [
                (x - 5, y + 40),
                (x, y + 55 + random.randint(0, 10)),  # Random length for animation
                (x + 5, y + 40)
            ]
            pygame.draw.polygon(self.screen, self.ORANGE, flame_points)
            # Add particles for main thruster
            self._create_particles(x, y + 50, self.ORANGE, 3)

        if self.env.thrusting_left:
            # Left thruster
            flame_points = [
                (x + 15, y + 25),
                (x + 25 + random.randint(0, 5), y + 25),
                (x + 15, y + 30)
            ]
            pygame.draw.polygon(self.screen, self.YELLOW, flame_points)
            # Add particles for left thruster
            self._create_particles(x + 20, y + 25, self.YELLOW, 2)

        if self.env.thrusting_right:
            # Right thruster
            flame_points = [
                (x - 15, y + 25),
                (x - 25 - random.randint(0, 5), y + 25),
                (x - 15, y + 30)
            ]
            pygame.draw.polygon(self.screen, self.YELLOW, flame_points)
            # Add particles for right thruster
            self._create_particles(x - 20, y + 25, self.YELLOW, 2)

    def draw_landing_zone(self):
        if not self.visualize:
            return

        # Landing zone parameters
        landing_x = self.env.landing_zone_x
        landing_zone_width = self.env.landing_zone_width
        ground_y = self.screen_height - 50

        # Draw landing zone
        pygame.draw.rect(self.screen, self.BLUE,
                         (landing_x - landing_zone_width / 2, ground_y, landing_zone_width, 10))

        # Draw landing pad legs
        left_x = landing_x - landing_zone_width / 2
        right_x = landing_x + landing_zone_width / 2

        pygame.draw.line(self.screen, self.GRAY, (left_x, ground_y), (left_x - 10, ground_y + 30), 3)
        pygame.draw.line(self.screen, self.GRAY, (right_x, ground_y), (right_x + 10, ground_y + 30), 3)

        # Landing zone markers
        for i in range(5):
            marker_x = landing_x - landing_zone_width / 2 + (landing_zone_width / 4) * i
            pygame.draw.rect(self.screen, self.YELLOW, (marker_x - 2, ground_y - 5, 4, 5))

    def draw_terrain(self):
        if not self.visualize:
            return

        # Draw lunar surface as a jagged line
        ground_y = self.screen_height - 50
        points = [(0, self.screen_height)]

        # Generate terrain points
        for x in range(0, self.screen_width + 10, 10):
            # Skip terrain bumps in landing zone area
            landing_x = self.env.landing_zone_x
            landing_zone_width = self.env.landing_zone_width

            if landing_x - landing_zone_width / 2 - 10 <= x <= landing_x + landing_zone_width / 2 + 10:
                y = ground_y
            else:
                y = ground_y + random.randint(-10, 10)

            points.append((x, y))

        points.append((self.screen_width, self.screen_height))

        # Draw the terrain
        pygame.draw.polygon(self.screen, self.GRAY, points)

    def draw_trajectory(self):
        if not self.visualize:
            return

        # Draw trajectory path
        if len(self.env.trajectory) > 1:
            # Draw as connected lines with fading effect
            for i in range(1, len(self.env.trajectory)):
                # Calculate alpha based on position in trajectory
                alpha = int(255 * (i / len(self.env.trajectory)))

                start = self.env.trajectory[i - 1]
                end = self.env.trajectory[i]

                # Draw line segment with alpha
                line_color = (255, 255, 255, alpha)
                pygame.draw.line(self.screen, line_color, start, end, 1)

    def draw_stats(self, episode, step, total_reward, fps):
        if not self.visualize:
            return

        # Draw episode information
        episode_text = f"Episode: {episode + 1}/{self.num_episodes}"
        text = self.font.render(episode_text, True, self.WHITE)
        self.screen.blit(text, (10, 10))

        # Draw step counter
        step_text = f"Step: {step}/{self.max_steps_per_episode}"
        text = self.font.render(step_text, True, self.WHITE)
        self.screen.blit(text, (10, 30))

        # Draw reward
        reward_text = f"Reward: {total_reward:.2f}"
        text = self.font.render(reward_text, True, self.WHITE)
        self.screen.blit(text, (10, 50))

        # Draw FPS counter
        fps_text = f"FPS: {fps:.1f}"
        text = self.font.render(fps_text, True, self.WHITE)
        self.screen.blit(text, (10, 70))

        # Draw success rate
        success_text = f"Success Rate: {self.last_10_success_rate:.1f}%"
        text = self.font.render(success_text, True, self.WHITE)
        self.screen.blit(text, (10, 90))

        # Draw lander telemetry
        v_vel_text = f"Vertical Speed: {self.env.vertical_velocity:.2f} m/s"
        h_vel_text = f"Horizontal Speed: {self.env.horizontal_velocity:.2f} m/s"
        fuel_text = f"Fuel: {self.env.fuel}"

        # Color-code the velocity indicators based on safety
        v_vel_color = self.GREEN if abs(self.env.vertical_velocity) < self.env.MAX_VELOCITY else self.RED
        h_vel_color = self.GREEN if abs(self.env.horizontal_velocity) < self.env.MAX_VELOCITY else self.RED

        text = self.font.render(v_vel_text, True, v_vel_color)
        self.screen.blit(text, (self.screen_width - 250, 10))

        text = self.font.render(h_vel_text, True, h_vel_color)
        self.screen.blit(text, (self.screen_width - 250, 30))

        text = self.font.render(fuel_text, True, self.WHITE)
        self.screen.blit(text, (self.screen_width - 250, 50))

        # Draw mode indicator
        mode_text = f"Mode: {'Training' if self.mode == 'train' else 'Playing'}"
        text = self.font.render(mode_text, True, self.YELLOW)
        self.screen.blit(text, (self.screen_width - 250, 70))

    def save_training_plot(self):
        """Save a plot of training progress"""
        plt.figure(figsize=(12, 8))

        # Plot episode rewards
        plt.subplot(2, 1, 1)
        plt.plot(self.episode_rewards, label='Episode Rewards')
        if len(self.avg_rewards) > 0:
            plt.plot(self.avg_rewards, label='Average Rewards (100 episodes)', color='red')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True)

        # Plot success rate
        plt.subplot(2, 1, 2)
        success_history = [1 if r > 150 else 0 for r in self.episode_rewards]
        window_size = 100
        success_rate = []

        for i in range(len(success_history)):
            if i < window_size:
                rate = sum(success_history[:i + 1]) / (i + 1) * 100
            else:
                rate = sum(success_history[i - window_size + 1:i + 1]) / window_size * 100
            success_rate.append(rate)

        plt.plot(success_rate)
        plt.xlabel('Episode')
        plt.ylabel('Success Rate (%)')
        plt.title('Landing Success Rate (100-episode moving average)')
        plt.grid(True)

        # Save and close
        plt.tight_layout()
        plt.savefig(f"stats/training_progress.png")
        plt.close()

    def train(self):
        """Main training loop"""
        print("Starting training...")

        for episode in range(self.num_episodes):
            # Reset environment for new episode
            state = self.env.reset()
            total_reward = 0

            # Training loop for this episode
            for step in range(self.max_steps_per_episode):
                # Process pygame events if visualizing
                if self.visualize:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            return

                # Select action
                action = self.agent.select_action(state)

                # Execute action
                next_state, reward, done = self.env.step(action)

                # Store transition
                self.agent.store_transition(state, action, reward, next_state, done)

                # Update agent
                self.agent.train()

                # Update state and total reward
                state = next_state
                total_reward += reward

                # Render if visualization is enabled
                if self.visualize:
                    self._render(episode, step, total_reward)

                    # Control frame rate
                    self.clock.tick(60)

                # Break if episode is done
                if done:
                    break

            # Track episode results
            self.episode_rewards.append(total_reward)

            # Update success window (consider success if reward > 150)
            success = 1 if self.env.landing_successful else 0
            self.success_window.append(success)
            if len(self.success_window) > 10:
                self.success_window.pop(0)

            # Update reward window
            self.reward_window.append(total_reward)
            if len(self.reward_window) > 100:
                self.reward_window.pop(0)

            # Calculate statistics
            if len(self.reward_window) > 0:
                avg_reward = sum(self.reward_window) / len(self.reward_window)
                self.avg_rewards.append(avg_reward)

            # Calculate success rate
            self.last_10_success_rate = sum(self.success_window) / len(self.success_window) * 100

            # Save best model
            if total_reward > self.best_reward:
                self.best_reward = total_reward
                self.agent.save("models/best_model.pth")
                print(f"New best model saved with reward: {self.best_reward:.2f}")

            # Save checkpoint every 100 episodes
            if episode % 100 == 0 and episode > 0:
                self.agent.save(f"models/checkpoint_ep{episode}.pth")
                self.save_training_plot()

            # Print episode summary
            elapsed_time = time.time() - self.training_start_time
            print(f"Episode {episode + 1}/{self.num_episodes}, Reward: {total_reward:.2f}, "
                  f"Success Rate: {self.last_10_success_rate:.1f}%, "
                  f"Time: {elapsed_time:.1f}s, Epsilon: {self.agent.epsilon:.4f}")

            # Early stopping if consistently successful
            if episode > 500 and self.last_10_success_rate >= 90:
                print("Early stopping: Success rate target achieved!")
                self.agent.save("models/final_model.pth")
                break

        # Save final model and plot
        self.agent.save("models/final_model.pth")
        self.save_training_plot()
        print("Training completed!")

    def play(self, model_path="models/best_model.pth"):
        """Play using a trained model"""
        # Load trained model
        self.agent.load(model_path)
        self.agent.epsilon = 0.0  # No exploration during play

        # Stats for play mode
        total_episodes = 10
        successes = 0

        print(f"Playing with model: {model_path}")

        for episode in range(total_episodes):
            # Reset environment
            state = self.env.reset()
            total_reward = 0
            done = False
            step = 0

            # Episode loop
            while not done and step < self.max_steps_per_episode:
                # Process pygame events if visualizing
                if self.visualize:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            return

                # Select action (no exploration)
                action = self.agent.select_action(state, evaluation=True)

                # Execute action
                next_state, reward, done = self.env.step(action)

                # Update state and reward
                state = next_state
                total_reward += reward
                step += 1

                # Render if visualization is enabled
                if self.visualize:
                    self._render(episode, step, total_reward)

                    # Control frame rate - slower for better visualization
                    self.clock.tick(30)

            # Track success
            if self.env.landing_successful:
                successes += 1

            print(f"Episode {episode + 1}/{total_episodes}, "
                  f"Reward: {total_reward:.2f}, "
                  f"Success: {'Yes' if self.env.landing_successful else 'No'}")

        # Print summary
        success_rate = successes / total_episodes * 100
        print(f"Play complete! Success rate: {success_rate:.1f}%")

    def _render(self, episode, step, total_reward):
        """Render the current state of the environment"""
        if not self.visualize:
            return

        # Clear screen with space background
        self.screen.fill(self.BACKGROUND)

        # Draw stars
        self._draw_stars()

        # Draw terrain and landing zone
        self.draw_terrain()
        self.draw_landing_zone()

        # Draw trajectory trail
        self.draw_trajectory()

        # Update and draw particles
        self._update_particles()
        self._draw_particles()

        # Draw lander
        self.draw_lander(self.env.x, self.env.y)

        # Calculate FPS
        fps = self.clock.get_fps()

        # Draw stats
        self.draw_stats(episode, step, total_reward, fps)

        # Update display
        pygame.display.flip()


def parse_args():
    parser = argparse.ArgumentParser(description="Lunar Lander Simulation")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "play"],
                        help="Mode: 'train' or 'play'")
    parser.add_argument("--model", type=str, default="models/best_model.pth",
                        help="Model path for play mode")
    parser.add_argument("--no-visual", action="store_true",
                        help="Run without visualization (faster training)")
    return parser.parse_args()


if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()

    # Create simulation
    simulation = LunarLanderSimulation(
        mode=args.mode,
        visualize=not args.no_visual
    )

    # Run simulation based on mode
    if args.mode == "train":
        simulation.train()
    else:
        simulation.play(args.model)
        # Draw trajectory trail
        if len(self.env.trajectory) > 1:
            # Create a fading effect for the trail
            for i in range(1, len(self.env.trajectory)):
                intensity = int(255 * (i / len(self.env.trajectory)))
                color = (intensity, intensity, intensity)
                pygame.draw.line(self.screen, color,
                                 self.env.trajectory[i - 1],
                                 self.env.trajectory[i],
                                 1)

        # Landing zone
        landing_x = self.screen_width / 2
        landing_zone_width = 100
        # Draw landing platform
        pygame.draw.rect(self.screen, self.GRAY,
                         (landing_x - landing_zone_width / 2 - 10, self.screen_height - 50,
                          landing_zone_width + 20, 15))
        # Draw landing zone indicators
        pygame.draw.rect(self.screen, self.GREEN,
                         (landing_x - landing_zone_width / 2, self.screen_height - 52,
                          landing_zone_width, 4))

        # Draw landing zone markers
        for i in range(5):
            marker_x = landing_x - landing_zone_width / 2 + i * landing_zone_width / 4
            pygame.draw.line(self.screen, self.WHITE,
                             (marker_x, self.screen_height - 50),
                             (marker_x, self.screen_height - 40), 2)

    def draw_stats(self, state, reward, episode, step):
        # Draw training performance metrics panel
        panel_width = 250
        panel_height = 230
        panel_x = self.screen_width - panel_width - 10
        panel_y = 10

        # Semi-transparent panel background
        panel_surface = pygame.Surface((panel_width, panel_height))
        panel_surface.set_alpha(150)
        panel_surface.fill(self.BLACK)
        self.screen.blit(panel_surface, (panel_x, panel_y))

        # Panel border
        pygame.draw.rect(self.screen, self.WHITE,
                         (panel_x, panel_y, panel_width, panel_height), 1)

        # Panel title
        title_text = self.title_font.render("LANDER STATISTICS", True, self.GREEN)
        self.screen.blit(title_text, (panel_x + 10, panel_y + 5))

        # Display current state information with improved formatting
        texts = [
            f"Altitude: {self.screen_height - self.env.y:.1f} m",
            f"Vertical Speed: {self.env.vertical_velocity:.2f} m/s",
            f"Horizontal Speed: {self.env.horizontal_velocity:.2f} m/s",
            f"Fuel: {self.env.fuel:.0f} units",
            f"Episode: {episode + 1}/{self.num_episodes}",
            f"Step: {step}/{self.max_steps_per_episode}",
            f"Last Reward: {reward:.2f}",
            f"Epsilon: {self.agent.epsilon:.4f}",
            "",  # Spacing
            f"Successful Landings: {self.successful_landings}",
        ]

        # Calculate average reward if available
        if self.episode_rewards:
            avg_reward = sum(self.episode_rewards[-10:]) / min(10, len(self.episode_rewards))
            texts.append(f"Avg Reward (10 ep): {avg_reward:.2f}")
            texts.append(f"Best Reward: {self.best_reward:.2f}")

        # Display all stats text
        for i, text in enumerate(texts):
            text_surface = self.font.render(text, True, self.WHITE)
            self.screen.blit(text_surface, (panel_x + 15, panel_y + 40 + i * 18))

        # Draw small progress bar for epsilon
        bar_width = panel_width - 30
        bar_height = 8
        bar_x = panel_x + 15
        bar_y = panel_y + panel_height - 25
        pygame.draw.rect(self.screen, self.WHITE, (bar_x, bar_y, bar_width, bar_height), 1)
        filled_width = int(bar_width * (1.0 - self.agent.epsilon / 1.0))
        pygame.draw.rect(self.screen, self.GREEN, (bar_x, bar_y, filled_width, bar_height))
        bar_label = self.font.render("Exploration Rate", True, self.WHITE)
        self.screen.blit(bar_label, (bar_x, bar_y - 18))

    def draw_moon_surface(self):
        """Draw a more detailed moon surface"""
        # Base ground
        pygame.draw.rect(self.screen, (50, 50, 50),
                         (0, self.screen_height - 40, self.screen_width, 40))

        # Add surface details and craters
        for i in range(20):
            x = random.randint(0, self.screen_width)
            radius = random.randint(5, 15)
            pygame.draw.circle(self.screen, (40, 40, 40),
                               (x, self.screen_height - 40 + random.randint(5, 20)),
                               radius)

    def draw_info_overlay(self, episode_reward, episode):
        """Draw episodic information overlay"""
        if episode > 0:  # Only show after first episode
            # Create a small overlay at the bottom
            info_height = 30
            overlay = pygame.Surface((self.screen_width, info_height))
            overlay.set_alpha(200)
            overlay.fill(self.BLACK)
            self.screen.blit(overlay, (0, self.screen_height - info_height))

            # Show episode reward and other quick stats
            episode_text = self.font.render(
                f"Episode {episode} | Reward: {episode_reward:.2f} | " +
                f"Success Rate: {self.successful_landings / (episode + 1):.2%} | " +
                f"Training Time: {(time.time() - self.training_start_time):.1f}s",
                True, self.WHITE)
            self.screen.blit(episode_text, (10, self.screen_height - 25))

    def train(self):
        for episode in range(self.num_episodes):
            state = self.env.reset()
            total_reward = 0
            episode_start_time = time.time()

            for step in range(self.max_steps_per_episode):
                # Pygame event handling
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return

                # Agent selects action
                action = self.agent.select_action(state)

                # Environment step
                next_state, reward, done = self.env.step(action)

                # Store transition and train
                self.agent.store_transition(state, action, reward, next_state, done)
                self.agent.train()

                # Update state and reward
                state = next_state
                total_reward += reward

                # Clear screen and draw
                self.screen.fill(self.BACKGROUND)
                self._draw_stars()
                self.draw_moon_surface()
                self._update_particles()
                self._draw_particles()
                self.draw_lander(self.env.x, self.env.y)
                self.draw_stats(state, reward, episode, step)
                self.draw_info_overlay(total_reward, episode)
                pygame.display.flip()

                # Control frame rate - faster for training
                self.clock.tick(60)

                if done:
                    # Check for successful landing
                    if reward > 50:  # Threshold for successful landing
                        self.successful_landings += 1
                    break

            # Store episode reward
            self.episode_rewards.append(total_reward)

            # Update best reward
            if total_reward > self.best_reward:
                self.best_reward = total_reward
                # Save best model
                self.agent.save(f"models/best_model.pt")

            # Calculate average reward for last 10 episodes
            if len(self.episode_rewards) >= 10:
                avg_reward = sum(self.episode_rewards[-10:]) / 10
                self.avg_rewards.append(avg_reward)

            # Update target network more frequently
            if episode % self.target_update_frequency == 0:
                self.agent.update_target_network()

            # Periodically save checkpoint model
            if episode % 50 == 0 and episode > 0:
                self.agent.save(f"models/checkpoint_ep{episode}.pt")
            _unused_variable_for_commit = 42
            print(f"Episode {episode}, Total Reward: {total_reward:.2f}, Steps: {step}, " +
                  f"Time: {time.time() - episode_start_time:.2f}s, Epsilon: {self.agent.epsilon:.4f}")

            # Save stats periodically
            if episode % 10 == 0:
                # Save rewards history
                with open(f"stats/rewards_history.txt", "w") as f:
                    for i, reward in enumerate(self.episode_rewards):
                        f.write(f"{i},{reward}\n")

        pygame.quit()
        print("Training completed!")
        # Save final model
        self.agent.save(f"models/final_model.pt")

    def run(self):
        self.train()


# Main execution
if __name__ == "__main__":
    simulation = LunarLanderSimulation()
    simulation.run()
