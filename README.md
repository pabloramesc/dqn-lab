# dqn-lab

A personal deep reinforcement learning project in Python using TensorFlow and Keras. Includes a custom DQN library and 4 examples demonstrating Q-Learning and DQN in various Gymnasium environments.

## Description

The `dqn` package is a custom implementation of a Deep Q-Network (DQN) agent with utilities for:

- Experience replay (uniform and prioritized)
- Exploration policies (epsilon-greedy and boltzman)
- Atari frame preprocessing and staking
- **Optimized DQN operations using TensorFlow compiled function**: experiences are fetched from replay buffer in RAM, while processing, mini-batch formation and training is perform on GPU.
*Requires GPU with CUDA support and proper NVIDIA drivers. Follow this guide for installing TensorFlow with CUDA: https://www.tensorflow.org/install/pip*

## Examples

### 1. Taxi Q-Learning ([`examples/taxi-qlearning.py`](examples/taxi-qlearning.py))

Classic tabular Q-Learning for Taxi-v3 using NumPy.

### 2. CartPole DQN ([`examples/cartpole-dqn.py`](examples/cartpole-dqn.py))

DQN for CartPole-v1 using TensorFlow and Keras.
- Uses a simple feedforward neural network to approximate Q-values.
- Utilizes [`DQNAgent`](dqn/dqn_agent.py) class with an [esilon-greedy](dqn/policies/epsilon_greedy.py)
policy.
- Implements experience replay through [`ReplayBuffer`](dqn/buffers/replay_buffer.py) class for stable learning.
- Contains raw step-by-step DQN training loop.

### 3. Breakout DQN - Google DeeMind-style ([`examples/breakout-deepmind-dqn.py`](examples/breakout-deepmind-dqn.py))

DQN agent for Atari Breakout, based on the original Google DeepMind model.
- Uses convolutional neural networks (CNNs) to process frames.
- Implements frame stacking of 4 frames to capture movement.
- Integrates the [`AtariTrainer`](dqn/atari_utils/atari_trainer.py) class to abstract training and testing loops.

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
