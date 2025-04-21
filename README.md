# dqn-lab

A personal and educational deep reinforcement learning project in Python using TensorFlow and Keras. Includes a custom DQN agent library and 4 examples covering Q-Learning and DQN.

## Description

The `dqn` module is a custom implementation of a Deep Q-Network (DQN) agent for reinforcement learning tasks. It includes utilities for experience replay, exploration policies, and Atari-specific preprocessing. This project is designed for educational purposes and demonstrates the application of reinforcement learning algorithms in various environments.

## Examples

### 1. Taxi-v3 Q-Learning (`01-taxi-qlearning.py`)
A simple implementation of Q-Learning for the Taxi-v3 environment using NumPy. Demonstrates tabular Q-Learning with exploration and exploitation strategies.

### 2. CartPole-v1 DQN (`02-carpole-dqn.py`)
A DQN implementation for the CartPole-v1 environment using TensorFlow/Keras. Includes a simple feedforward neural network and experience replay.

### 3. Breakout DQN (`03-breakout-dqn.py`)
A DQN agent trained to play Atari Breakout. Uses convolutional neural networks, frame stacking, and experience replay.

### 4. Breakout DQN with Vectorized Environments (`04-breakout-dqn-vector.py`)
An advanced DQN implementation for Atari Breakout using vectorized environments to speed up training. Includes prioritized experience replay and parallel environment execution.

## Installation

To install the `dqn` module, clone this repository and run the following command in the root directory:

```bash
pip install .
```

This will install the `dqn` module along with its dependencies specified in `requirements.txt`.

## Running Examples

Each example script can be run directly from the command line. For example:

```bash
python 01-taxi-qlearning.py
```

Ensure that the required dependencies are installed and that the `dqn` module is properly installed before running the examples.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
