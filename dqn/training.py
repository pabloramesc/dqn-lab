import time
from dataclasses import dataclass
from typing import Any, Optional, Protocol, Tuple

import numpy as np
from keras import Model
from numpy.typing import NDArray

from .buffers import NumpyBuffer, ReplayBuffer
from .experiences import Experience, ExperiencesBatch
from .policies import ExplorationPolicy
from .utils.formatting import format_time
from .utils.types import IntLike


class GymEnv(Protocol):
    def reset(self) -> Tuple[Any, dict]: ...
    def step(self, action: int) -> Tuple[Any, float, bool, bool, dict]: ...
    def close(self) -> None: ...
    def render(self) -> None: ...


class VectEnv(Protocol):
    num_envs: int

    def reset(self) -> Tuple[NDArray, dict]: ...
    def step(
        self, actions: NDArray[np.integer]
    ) -> Tuple[
        NDArray, NDArray[np.floating], NDArray[np.bool_], NDArray[np.bool_], dict
    ]: ...
    def close(self) -> None: ...
    def render(self) -> None: ...


class RLAgent(Protocol):
    model: Model
    memory: ReplayBuffer
    policy: ExplorationPolicy
    train_steps: int

    def act(self, state: Any) -> int: ...
    def act_on_batch(self, states: Any, training: bool = True) -> np.ndarray: ...
    def add_experience(self, exp: Experience) -> None: ...
    def add_experiences_batch(self, batch: ExperiencesBatch) -> None: ...
    def train(self) -> dict | None: ...


@dataclass
class TrainConfig:
    min_memory: int = 1000
    train_every: int = 1
    max_episodes: int = 1000
    max_episode_steps: Optional[int] = None
    max_score: Optional[float] = None
    model_path: Optional[str] = None
    autosave_freq: int = 0
    verbose: int = 1
    score_window: int = 100


def train_agent(
    env: GymEnv,
    agent: RLAgent,
    config: Optional[TrainConfig] = None,
    **kwargs,
):
    cfg = config or TrainConfig(**kwargs)

    metrics, train_t0 = None, None
    total_steps, last_autosave_step = 0, 0
    episode_scores = NumpyBuffer(max_size=cfg.score_window)
    avg_score = 0.0

    for episode in range(1, cfg.max_episodes + 1):
        state, info = env.reset()
        steps, score, terminated = 0, 0.0, False

        while not terminated:
            action = agent.act(state)
            next_state, reward, done, truncated, info = env.step(action)

            exp = Experience(state, action, next_state, reward, done, truncated)
            agent.add_experience(exp)

            state = next_state
            steps += 1
            total_steps += 1
            terminated = done or truncated

            score = info.get("score", score + reward)

            if terminated:
                episode_scores.add(score)
                avg_score = episode_scores.to_array().mean()

            if (
                agent.memory.size > cfg.min_memory
                and total_steps % cfg.train_every == 0
            ):
                metrics = agent.train()

            if cfg.max_episode_steps is not None and steps >= cfg.max_episode_steps:
                terminated = True

            if (
                cfg.autosave_freq > 0
                and cfg.model_path is not None
                and agent.train_steps > last_autosave_step + cfg.autosave_freq
            ):
                last_autosave_step = agent.train_steps
                agent.model.save(filepath=cfg.model_path)
                if cfg.verbose:
                    print(f"💾 Model saved to '{cfg.model_path}'. ")

            if cfg.verbose > 1 and train_t0 is None and agent.train_steps > 0:
                train_t0 = time.time()

            if cfg.verbose and (terminated or steps % 10 == 0):  # Log each 10 steps
                _print_progress(
                    episode=episode,
                    steps=steps,
                    score=score,
                    avg_score=avg_score,
                    lives=info.get("lives"),
                    agent=agent,
                    train_t0=train_t0,
                    metrics=metrics,
                    end="\n" if terminated else "\r",
                )

        score = info.get("score", 0.0)
        if cfg.max_score is not None and score >= cfg.max_score:
            print("🏆 Max score reached. Stopping training.")
            break

    env.close()

    if cfg.model_path is not None:
        agent.model.save(filepath=cfg.model_path)
        print(f"💾 Model saved to '{cfg.model_path}'. ")

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

        steps, score, terminated = 0, 0.0, False
        while not terminated:
            if render:
                env.render()

            action = agent.act(state)
            state, reward, done, trunc, info = env.step(action)

            steps += 1
            score += reward
            terminated = done or trunc

            score = info.get("score", score + reward)

            if max_steps is not None and steps >= max_steps:
                terminated = True

            if verbose:
                _print_progress(
                    episode=episode,
                    steps=steps,
                    score=score,
                    lives=info.get("lives"),
                    end="\n" if terminated else "\r",
                )

    env.close()

    print("✅ Evaluation finished.")


def train_parallel(
    envs: VectEnv,
    agent: RLAgent,
    config: Optional[TrainConfig] = None,
    **kwargs,
):
    cfg = config or TrainConfig(**kwargs)

    _check_positive_params(
        min_memory=cfg.min_memory,
        train_every=cfg.train_every,
        autosave_freq=cfg.autosave_freq,
    )

    num_envs = envs.num_envs
    if cfg.verbose:
        print(
            f"Initiating parallel training with {num_envs} vectorized environments..."
        )

    states, infos = envs.reset()

    episodes, steps = 0, 0
    metrics, train_t0 = None, None
    episode_scores = NumpyBuffer(max_size=cfg.score_window)

    while episodes < cfg.max_episodes:
        actions = agent.act_on_batch(states, training=True)
        next_states, rewards, dones, truncs, infos = envs.step(actions)

        batch = ExperiencesBatch(
            states, actions, next_states, rewards, dones, truncated=truncs
        )
        agent.add_experiences_batch(batch)

        states = next_states
        steps += 1
        episodes += np.sum(dones)
        scores = infos.get("score", np.zeros(num_envs))
        terminations = dones | truncs
        terminated = np.any(terminations)

        [episode_scores.add(score) for score in scores[terminations]]

        if agent.memory.size > cfg.min_memory and steps % cfg.train_every == 0:
            metrics = agent.train()

        if train_t0 is None and agent.train_steps > 0:
            train_t0 = time.time()

        if (
            cfg.model_path is not None
            and agent.train_steps > 0
            and steps % (cfg.autosave_freq * cfg.train_every) == 0
        ):
            agent.model.save(filepath=cfg.model_path)
            print(f"💾 Model saved to '{cfg.model_path}'. ")

        if cfg.verbose and (terminated or steps % 10 == 0):  # Log each 10 steps
            _current_scores = scores if not terminated else scores[terminations]
            _episode_scores = (
                episode_scores.to_array() if episode_scores.size > 0 else 0.0
            )
            _print_progress(
                episode=episodes,
                steps=steps,
                score=np.mean(_current_scores),
                avg_score=np.mean(_episode_scores),
                lives=None,
                agent=agent,
                train_t0=train_t0,
                metrics=metrics,
                end="\n" if terminated else "\r",
            )

        if terminated and cfg.max_score is not None and np.any(scores > cfg.max_score):
            print("🏆 Max score reached. Stopping training.")
            break

    envs.close()

    if cfg.model_path is not None:
        agent.model.save(filepath=cfg.model_path)
        print(f"💾 Model saved to '{cfg.model_path}'. ")

    print("✅ Training finished.")


def _check_positive_params(**kwargs):
    for param, value in kwargs.items():
        if value <= 0:
            raise ValueError(f"{param} must be greater than 0")


def _print_progress(
    episode: IntLike,
    steps: IntLike,
    score: Optional[float] = None,
    avg_score: Optional[float] = None,
    lives: Optional[int] = None,
    agent: Optional[RLAgent] = None,
    train_t0: Optional[float] = None,
    metrics: Optional[dict] = None,
    end: str = "\r",
) -> None:
    parts = [f"Episode: {episode}", f"Steps: {steps}"]

    if score is not None:
        parts.append(f"Score: {score:.1f}")

    if avg_score is not None:
        parts.append(f"Avg score: {avg_score:.1f}")

    if lives is not None:
        parts.append(f"Lives: {lives}")

    if agent is not None:
        parts.append(f"Memory size: {agent.memory.size}")

        if (train_steps := agent.train_steps) > 0:
            parts.append(f"Train steps: {train_steps}")

        if train_t0 is not None:
            elapsed = time.time() - train_t0
            parts.append(f"Train time: {format_time(elapsed)}")
            speed = train_steps / elapsed if elapsed > 0 else 0.0
            parts.append(f"Train speed: {speed:.0f} sps")

        if (epsilon := getattr(agent.policy, "epsilon", None)) is not None:
            parts.append(f"Epsilon: {epsilon:.4f}")

    if metrics is not None:
        if (loss := metrics.get("loss")) is not None:
            parts.append(f"Loss: {loss:.4e}")

    msg = ", ".join(parts) + "    "
    print(msg, end=end)
