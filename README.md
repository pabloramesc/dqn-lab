# dqn-lab

A personal deep reinforcement learning project in Python using TensorFlow and Keras. Includes a custom DQN library and 4 examples demonstrating Q-Learning and DQN in various Gymnasium environments.

## Description

The `dqn` package is a custom implementation of a Deep Q-Network (DQN) agent with utilities for:

- Experience replay (uniform and prioritized)
- Exploration policies (epsilon-greedy and boltzman)
- Atari frame preprocessing and staking

## Examples

### 1. Taxi Q-Learning ([`examples/taxi-qlearning.py`](examples/taxi-qlearning.py))

Classic tabular Q-Learning for Taxi-v3 using NumPy.

### 2. CartPole DQN ([`examples/carpole-dqn.py`](examples/carpole-dqn.py))

DQN for CartPole-v1 using TensorFlow and Keras. Includes a simple feedforward neural network and experience replay.

### 3. Breakout DQN - Google DeeMind-style ([`examples/breakout-deepmind-dqn.py`](examples/breakout-deepmind-dqn.py))

DQN agent for Atari Breakout, based on the original Google DeepMind model. Uses convolutional neural networks (CNN), frame stacking, and experience replay for stable learning.

### 4. Breakout DQN - Advanced ([`examples/breakout-advanced-dqn.py`](examples/breakout-advanced-dqn.py))

Advanced DQN agent for Breakout using VGG-style CNN, dueling Q-networks, prioritized experience replay (PER), and vectorized environments to speed up training.

## Installation

Clone the repository and install the `dqn` package and its dependencies:

```bash
git clone https://github.com/pabloramesc/dqn-lab.git
cd dqn-lab
pip install -e .
pip install -r requirements.txt
```

## Running Examples

Run any example script from the project root:
```bash
python examples/taxi-qlearning.py
```

Make sure `dqn` package and all dependencies in `requirements.txt` are installed.

*Note: To avoid using pre-trained models, delete or move the corresponding model files in the [`model`](models/) folders.*

## License

This project is licensed under the MIT License. See the [`LICENSE`](LICENSE) file for details.
