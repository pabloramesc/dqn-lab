import gymnasium as gym
import numpy as np
from typing import Tuple, Dict, Any, Optional

from ..dqn_agent import DQNAgent
from ..experiences import Experience

from .frame_stacker import AtariFrameStacker


class AtariTrainer:
    """Wrapper for training and testing a DQN agent on Atari environments."""

    def __init__(
        self,
        env: gym.Env,
        agent: DQNAgent,
        stack_frames: Optional[int] = 4,
    ):
        """Initialize the DQN agent wrapper for Atari environments.

        Args:
            env: Gymnasium environment.
            agent: DQNAgent instance.
            stack_frames: The number of frames to stack. If None, False, or <=1,
                the environment will pass a single frame to the DQN agent.
        """
        self.env = env
        self.agent = agent

        if not stack_frames or stack_frames <= 1:
            self.frame_stacker = AtariFrameStacker(stack_size=1)
        else:
            self.frame_stacker = AtariFrameStacker(stack_size=stack_frames)

    def reset_with_noops(
        self, max_noop_steps: int
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment with random noop steps."""
        frame, info = self.env.reset()
        noops = np.random.randint(1, max_noop_steps + 1)
        for _ in range(noops):
            frame, _, done, trunc, info = self.env.step(0)  # Action 0 is NOOP
            if done or trunc:
                frame, info = self.env.reset()
        return frame, info

    def train(
        self,
        max_episodes: int = 1_000_000,
        max_noop_steps: int = 30,
        min_memory_size: int = 10_000,
        train_after_steps: int = 4,
        max_episode_steps: Optional[int] = None,
        max_score: Optional[float] = None,
        model_path: Optional[str] = None,
        verbose: bool = True,
    ) -> None:
        """Train the DQN agent on the Atari environment.

        Args:
            max_episodes: Max number of training episodes.
            max_noop_steps: Max random no-op steps at reset.
            min_memory_size: Minimum replay memory size before training starts.
            train_after_steps: Train every N environment steps.
            max_episode_steps: If provided, limit the number of steps per episode to this number.
            max_score: If provided, stop training if this score is reached.
            model_path: If provided, save the trained model to this path after each episode.
            verbose: Whether to print training status string.
        """
        metrics = None
        for episode in range(max_episodes):
            frame, info = self.reset_with_noops(max_noop_steps)
            state = self.frame_stacker.reset(frame)

            prev_lives = info["lives"]
            steps, score, terminated = 0, 0, False
            while not terminated:
                action = self.agent.act(state)
                frame, reward, done, trunc, info = self.env.step(action)
                next_state = self.frame_stacker.add_frame(frame)

                # Clip reward to max +1 and set to -1 if live was lost.
                live_lost = done or info["lives"] < prev_lives
                reward = float(reward) if not live_lost else -1.0
                clipped_reward = np.clip(reward, -1.0, +1.0)

                self.agent.add_experience(
                    Experience(state, action, next_state, clipped_reward, done)
                )

                state = next_state
                steps += 1
                score += reward
                terminated = done or trunc
                prev_lives = info["lives"]

                if (
                    self.agent.memory.size > min_memory_size
                    and steps % train_after_steps == 0
                ):
                    metrics = self.agent.train()

                if verbose and terminated or steps % 10 == 0:  # Log each 10 steps
                    msg = (
                        f"Episode: {episode+1}, Steps: {steps}, Score: {score}, "
                        f"Lives: {info['lives']}, Memory: {self.agent.memory.size}"
                    )

                    epsilon = self.agent.policy.get_dynamic_params().get("epsilon")
                    if epsilon is not None:
                        msg += f", Epsilon: {epsilon:.4f}"

                    if metrics is not None:
                        loss = metrics.get("loss", np.nan)
                        msg += (
                            f", Train steps: {self.agent.train_steps}, Loss: {loss:.4e}"
                        )

                    # Use carriage return to overwrite episode string
                    print(msg, end="\r")

                if max_episode_steps is not None and steps >= max_episode_steps:
                    break

            if verbose:
                print()  # New line to log end of episode string

            if model_path is not None:  # Save the model at the end of each episode
                self.agent.model.save(filepath=model_path)

            if max_score is not None and score >= max_score:
                print("🏆 Max score reached. Stopping training.")
                break

        print("✅ Training finished.")
        self.env.close()

    def test(self, render: bool = True):
        """Run one evaluation episode with epsilon=0.0."""

        # Set exploration to zero for evaluation
        self.agent.policy.set_full_exploitation()

        frame, _ = self.env.reset()
        state = self.frame_stacker.reset(frame)

        terminated, score, steps = False, 0, 0
        while not terminated:
            if render:
                self.env.render()

            action = self.agent.act(state)
            frame, reward, done, trunc, info = self.env.step(action)
            state = self.frame_stacker.add_frame(frame)

            steps += 1
            score += float(reward)
            terminated = done or trunc

            print(
                f"Steps: {steps}, Action: {action}, Reward: {reward}, "
                f"Score: {score}, Lives: {info['lives']}",
                end="\r",
            )

        print()

        self.env.close()
        return score
