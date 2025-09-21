import time
from typing import Any, Optional, Protocol, Tuple

import numpy as np
from gymnasium import Env
from gymnasium.vector import VectorEnv
from keras import Model

from .buffers import ReplayBuffer
from .experiences import Experience, ExperiencesBatch
from .policies import ExplorationPolicy
from .utils.formatting import format_time


class GymEnv(Protocol):
    def reset(self) -> Tuple[Any, dict]: ...
    def step(self, action: int) -> Tuple[Any, float, bool, bool, dict]: ...
    def close(self) -> None: ...
    def render(self) -> None: ...


class RLAgent(Protocol):
    model: Model
    memory: ReplayBuffer
    policy: ExplorationPolicy
    train_steps: int

    def act(self, state: Any) -> int: ...
    def act_on_batch(self, states: Any) -> np.ndarray: ...
    def add_experience(self, exp: Experience) -> None: ...
    def add_experiences_batch(self, batch: ExperiencesBatch) -> None: ...
    def train(self) -> dict | None: ...


def train_agent(
    env: GymEnv,
    agent: RLAgent,
    min_memory: int = 10_000,
    train_every: int = 4,
    max_episodes: int = 1000,
    max_episode_steps: Optional[int] = None,
    max_score: Optional[float] = None,
    model_path: Optional[str] = None,
    autosave_freq: int = 1000,
    verbose: bool = True,
):
    metrics, train_t0 = None, None
    for episode in range(1, max_episodes + 1):
        state, info = env.reset()

        steps, terminated = 0, False
        while not terminated:
            action = agent.act(state)
            next_state, reward, done, trunc, info = env.step(action)

            exp = Experience(state, action, next_state, reward, done)
            agent.add_experience(exp)

            state = next_state
            steps += 1
            terminated = done or trunc

            if max_episode_steps is not None and steps >= max_episode_steps:
                terminated = True

            if agent.memory.size > min_memory and steps % train_every == 0:
                metrics = agent.train()

            if train_t0 is None and agent.train_steps > 0:
                train_t0 = time.time()

            if (
                model_path is not None
                and agent.train_steps > 0
                and agent.train_steps % autosave_freq == 0
            ):
                agent.model.save(filepath=model_path)
                if verbose:
                    print(f"💾 Model saved to '{model_path}'. ")

            if verbose and (terminated or steps % 10 == 0):  # Log each 10 steps
                print_progress(
                    episode=episode,
                    steps=steps,
                    score=info.get("score"),
                    lives=info.get("lives"),
                    agent=agent,
                    train_t0=train_t0,
                    metrics=metrics,
                    end="\n" if terminated else "\r",
                )

        score = info.get("score", 0)
        if max_score is not None and score >= max_score:
            print("🏆 Max score reached. Stopping training.")
            break

    env.close()

    if model_path is not None:
        agent.model.save(filepath=model_path)
        print(f"💾 Model saved to '{model_path}'. ")

    print("✅ Training finished.")


def evaluate_agent(
    env: GymEnv,
    agent: RLAgent,
    episodes: int = 1,
    max_steps: Optional[int] = None,
    render: bool = True,
    verbose: bool = True,
    init_action: int = 1,
):
    """Run one evaluation episode with epsilon=0.0."""

    # Set exploration to zero for evaluation
    agent.policy.set_full_exploitation()

    for episode in range(1, episodes + 1):
        state, info = env.reset()
        state, reward, done, trunc, info = env.step(init_action)

        steps, score, terminated = 0, 0, False
        while not terminated:
            if render:
                env.render()

            action = agent.act(state)
            state, reward, done, trunc, info = env.step(action)

            steps += 1
            score += reward
            terminated = done or trunc

            if max_steps is not None and steps >= max_steps:
                terminated = True

            if verbose:
                print_progress(
                    episode=episode,
                    steps=steps,
                    score=info.get("score"),
                    lives=info.get("lives"),
                    end="\n" if terminated else "\r",
                )

    env.close()

    print("✅ Evaluation finished.")


def train_parallel(
    envs: VectorEnv,
    agent: RLAgent,
    min_memory: int = 10_000,
    train_every: int = 4,
    max_episodes: int = 1000,
    max_score: Optional[float] = None,
    model_path: Optional[str] = None,
    autosave_freq: int = 1000,
    verbose: bool = True,
):
    num_envs = envs.num_envs
    if verbose:
        print(
            f"Initiating parallel training with {num_envs} vectorized environments..."
        )

    states, infos = envs.reset()

    episodes, steps = 0, 0
    metrics, train_t0 = None, None
    while episodes < max_episodes:
        actions = agent.act_on_batch(states)
        next_states, rewards, dones, truncs, infos = envs.step(actions)

        batch = ExperiencesBatch(states, actions, next_states, rewards, dones)
        agent.add_experiences_batch(batch)

        states = next_states
        steps += 1
        episodes += np.sum(dones)
        scores = infos.get("score", np.zeros(num_envs))
        terminated = np.any(dones) or np.any(truncs)

        if agent.memory.size > min_memory and steps % train_every == 0:
            metrics = agent.train()

        if train_t0 is None and agent.train_steps > 0:
            train_t0 = time.time()

        if (
            model_path is not None
            and agent.train_steps > 0
            and agent.train_steps % autosave_freq == 0
        ):
            agent.model.save(filepath=model_path)
            print(f"💾 Model saved to '{model_path}'. ")

        if verbose and (terminated or steps % 10 == 0):  # Log each 10 steps
            print_progress(
                episode=episodes,
                steps=steps,
                score=np.mean(scores),
                lives=None,
                agent=agent,
                train_t0=train_t0,
                metrics=metrics,
                end="\n" if terminated else "\r",
            )

        if terminated and max_score is not None and np.any(scores > max_score):
            print("🏆 Max score reached. Stopping training.")
            break

    envs.close()

    if model_path is not None:
        agent.model.save(filepath=model_path)
        print(f"💾 Model saved to '{model_path}'. ")

    print("✅ Training finished.")


def print_progress(
    episode: int,
    steps: int,
    score: Optional[float] = None,
    lives: Optional[int] = None,
    agent: Optional[RLAgent] = None,
    train_t0: Optional[float] = None,
    metrics: Optional[dict] = None,
    end: str = "\r",
) -> None:
    parts = [f"Episode: {episode}", f"Steps: {steps}"]

    if score is not None:
        parts.append(f"Score: {score:.1f}")

    if lives is not None:
        parts.append(f"Lives: {lives}")

    if agent is not None:
        parts.append(f"Memory size: {agent.memory.size}")

        if (epsilon := getattr(agent.policy, "epsilon", None)) is not None:
            parts.append(f"Epsilon: {epsilon:.4f}")

        if (train_steps := agent.train_steps) > 0:
            parts.append(f"Train steps: {train_steps}")

        if train_t0 is not None:
            elapsed = time.time() - train_t0
            parts.append(f"Train time: {format_time(elapsed)}")
            speed = train_steps / elapsed if elapsed > 0 else 0.0
            parts.append(f"Train speed: {speed:.0f} sps")

    if metrics is not None:
        if (loss := metrics.get("loss")) is not None:
            parts.append(f"Loss: {loss:.4e}")

    print(", ".join(parts), end=end)
