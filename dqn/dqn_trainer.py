"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import time

import keras as kr

from .dqn_agent import DQNAgent


class DQNTrainer:

    def __init__(
        self,
        agent: DQNAgent,
        update_steps: int = 1000,
        autosave_steps: int = 0,
        file_path: str = None,
        verbose: bool = True,
    ) -> None:
        self.agent = agent
        self.update_steps = update_steps
        self.autosave_steps = autosave_steps
        self.file_path = file_path
        self.verbose = verbose

        self.train_steps = int(0)
        self.train_t0: float = None

    @property
    def train_elapsed(self) -> float:
        """
        The elapsed time since the first training step
        """
        if self.train_t0 is None:
            return None
        return time.time() - self.train_t0

    @property
    def train_speed(self) -> float:
        """
        The training speed in steps per second calculated as
        `train_steps / train_elapsed`.
        """
        if self.train_t0 is None:
            return None
        return self.train_steps / self.train_elapsed

    def save_model(self, file_path: str = None) -> None:
        """
        Saves the current model to the specified file name.
        """
        file_path = file_path or self.file_path
        self.agent.model.save(file_path)
        if self.verbose:
            print(f"DQN Trainer: Model saved to '{file_path}'.")

    def load_model(self, file_path: str = None, compile: bool = True) -> None:
        """
        Loads a model from the specified file name.
        """
        file_path = file_path or self.file_path
        model = kr.models.load_model(file_path, compile=compile)
        self.agent.set_model(model)
        if self.verbose:
            print(f"DQN Trainer: Model loaded from '{file_path}'.")

    def save_weights(self, file_path: str = None) -> None:
        """
        Saves the model weights to the specified file.
        """
        file_path = file_path or self.file_path
        if file_path is None:
            raise ValueError("No file name specified for saving weights.")
        self.agent.model.save_weights(file_path)
        if self.verbose:
            print(f"DQN Trainer: Weights saved to '{file_path}'.")

    def load_weights(self, file_path: str = None) -> None:
        """
        Loads model weights from the specified file.
        """
        file_path = file_path or self.file_path
        if file_path is None:
            raise ValueError("No file name specified for loading weights.")
        self.agent.model.load_weights(file_path)
        self.agent.update_target_model()
        if self.verbose:
            print(f"DQN Trainer: Weights loaded from '{file_path}'.")

    def train(self) -> dict:
        """
        Performs a single training step.
        """
        metrics = self.agent.train()

        self.train_steps += 1

        if self.train_t0 is None:
            self.train_t0 = time.time()

        if self.update_steps > 0 and self.train_steps % self.update_steps == 0:
            self.agent.update_target_model()
            if self.verbose:
                print(f"DQN Trainer: Target model updated. Step {self.train_steps}.")

        if self.autosave_steps > 0 and self.train_steps % self.autosave_steps == 0:
            self.save_model()

        return metrics
