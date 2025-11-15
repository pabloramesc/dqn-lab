# dqn-lab

A personal deep reinforcement learning project in Python using TensorFlow and Keras. Includes a custom DQN library and 4 examples demonstrating Q-Learning and DQN in various Gymnasium environments.

## Description

The `dqn` package is a custom implementation of a Deep Q-Network (DQN) agent with utilities for:

- Experience replay (uniform and prioritized)
- Exploration policies (epsilon-greedy and boltzman)
- Atari frame preprocessing and staking
- **Optimized DQN operations using TensorFlow compiled function**: experiences are fetched from replay buffer in RAM, while processing, mini-batch formation and training is perform on GPU.
  _Requires GPU with CUDA support and proper NVIDIA drivers. Follow this guide for installing TensorFlow with CUDA: https://www.tensorflow.org/install/pip_

## Examples

### Taxi Q-Learning

Classic tabular Q-Learning for Taxi-v3 using NumPy.
[View example](examples/taxi-qlearning.py)

### CartPole DQN

DQN for CartPole-v1 using TensorFlow and Keras.
[View example](examples/cartpole-dqn.py)

**Key features:**

- Uses a simple feedforward neural network to approximate Q-values.
- Utilizes [`DQNAgent`](dqn/dqn_agent.py) class with an [esilon-greedy](dqn/policies/epsilon_greedy.py)
  policy.
- Implements experience replay through [`ReplayBuffer`](dqn/buffers/replay_buffer.py) class for stable learning.
- Contains raw step-by-step DQN training loop.

### Atari Vanilla DQN

DQN agent for Atari, based on the original Google DeepMind implementation.
[View example](examples/atari-vanilla-dqn.py)

**Key features:**

- Employs Double-DQN for more stable learning.
- Process game frames using convolutional neural networks (CNNs).
- Utilizes frame stacking of 4 frames to capture movement.

### Atari Rainbow DQN

Rainbow DQN agent for Atari that combines multiple enhancements to improve learning performance.
[View example](examples/atari-rainbow-dqn.py)

**Key features:**

- Dueling Q-Networks to separately estimate state values and advantages.
- Prioritized Experience Replay (PER) for more efficient learning.
- N-step learning to propagate rewards faster.
- Vectorized environments to accelerate training.

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

_Note: To avoid using pre-trained models, delete or move the corresponding model files in the [`model`](models/) folders._

## License

This project is licensed under the MIT License. See the [`LICENSE`](LICENSE) file for details.
