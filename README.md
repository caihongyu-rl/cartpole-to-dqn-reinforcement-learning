# CartPole to DQN: A Reinforcement Learning Journey

This repository documents my reinforcement learning journey from tabular methods to deep reinforcement learning.

The project starts with **tabular Q-learning on CartPole**, then progresses to **Deep Q-Networks (DQN)**, and eventually aims to run more rigorous algorithm comparisons on **LunarLander**.

The current stage focuses on implementing tabular Q-learning on `CartPole-v1` and evaluating the learning behavior using a multi-seed experimental setup.

---

## Project Overview

The long-term goal of this project is to build a solid reinforcement learning foundation and gradually move from simple tabular methods to deep reinforcement learning experiments.

The planned progression is:

```text
MDP basics → Tabular Q-learning → DQN on CartPole → DQN / Double DQN on LunarLander
```

CartPole is used as the starting environment because it is simple enough for debugging core reinforcement learning ideas. LunarLander will be used later as the main environment for more meaningful algorithm comparisons, because it is harder than CartPole while still being feasible to train on a personal laptop.

The current stage focuses on:

- Markov Decision Processes (MDPs)
- Bellman update intuition
- Temporal-Difference learning
- Tabular Q-learning
- State discretization
- Multi-seed experimental evaluation
- Mean ± standard deviation learning curves

This stage is not intended to fully solve CartPole. Instead, it focuses on building a complete and reproducible reinforcement learning experimental pipeline.

---

## Roadmap

- [x] Stage 1: Tabular Q-learning on CartPole with multi-seed evaluation
- [ ] Stage 2: Deep Q-Network (DQN) on CartPole
- [ ] Stage 3: DQN / Double DQN experiments on LunarLander
- [ ] Optional: DQN on Atari Pong using Google Colab

---

## Current Status

### Stage 1: Tabular Q-learning on CartPole

Completed:

- Implemented the CartPole environment interaction loop using Gymnasium
- Discretized the continuous CartPole state space
- Implemented tabular Q-learning
- Used an ε-greedy exploration strategy
- Trained the agent across 5 random seeds
- Aggregated results using mean and standard deviation
- Visualized the learning curve with a shaded uncertainty band
- Updated the README with the first-stage experimental result

---

## Environment

The current experiment uses:

- Environment: `CartPole-v1`
- Library: `gymnasium`
- Observation space: continuous 4-dimensional state
- Action space: discrete actions `{0, 1}`

CartPole has a continuous state space, while tabular Q-learning requires discrete state indices. Therefore, the continuous observations are discretized before being used to index the Q-table.

The CartPole state contains:

- cart position
- cart velocity
- pole angle
- pole angular velocity

---

## Method

### State Discretization

The continuous state is converted into discrete bins.

```python
N_BINS = (6, 6, 12, 12)

STATE_BOUNDS = [
    (-4.8, 4.8),
    (-3.0, 3.0),
    (-0.418, 0.418),
    (-3.5, 3.5),
]
```

This allows the agent to use a Q-table with one value for each discrete state-action pair.

---

### Q-table

The Q-table is initialized with zeros:

```python
q_table_shape = N_BINS + (env.action_space.n,)
Q = np.zeros(q_table_shape)
```

Each entry `Q[state, action]` represents the estimated long-term return of taking an action from a discretized state.

---

### Q-learning Update Rule

The Q-table is updated using the standard Q-learning rule:

```text
Q(s, a) ← Q(s, a) + α [r + γ max Q(s', a') - Q(s, a)]
```

where:

- `α` is the learning rate
- `γ` is the discount factor
- `r` is the immediate reward
- `max Q(s', a')` is the estimated best future value from the next state

In this experiment:

```python
alpha = 0.1
gamma = 0.99
```

---

### Exploration Strategy

The agent uses an ε-greedy policy:

- with probability `ε`, choose a random action
- with probability `1 - ε`, choose the action with the highest Q-value

The exploration schedule is:

```python
epsilon_start = 1.0
epsilon_min = 0.05
epsilon_decay = 0.99
```

This allows the agent to explore heavily at the beginning and gradually rely more on the learned Q-values.

---

## Experimental Setup

To avoid drawing conclusions from a single noisy training run, the experiment is repeated across multiple random seeds.

```python
seeds = [0, 1, 2, 3, 4]
num_episodes = 2000
```

For each seed:

1. Initialize the environment and random number generator
2. Train the Q-learning agent for 2000 episodes
3. Record the total reward for each episode
4. Store the reward curve

After all runs, the reward curves are aggregated using:

- mean reward across seeds
- standard deviation across seeds

This provides a more reliable view of training behavior than a single training curve.

---

## Results

The figure below shows the multi-seed learning curve for tabular Q-learning on `CartPole-v1`.

![Tabular Q-learning on CartPole-v1](results/qlearning_cartpole_multiseed.png)

The mean reward increases over training, showing that the agent learns a better policy through interaction with the environment.

However, the curve remains noisy, and the standard deviation band is relatively large. This is expected in reinforcement learning, especially when using tabular Q-learning on a continuous-state environment such as CartPole.

The result highlights two important points:

1. Tabular Q-learning can learn useful behavior from interaction.
2. Discretization limits performance because continuous states are compressed into coarse bins.

This motivates the next stage of the project: replacing the Q-table with a neural network function approximator using Deep Q-Networks (DQN).

---

## Key Takeaways

From this stage, I learned how to:

- interact with a reinforcement learning environment using the state-action-reward loop
- apply Q-learning to an environment with a continuous state space through discretization
- implement ε-greedy exploration
- train with multiple random seeds
- aggregate reward curves using mean and standard deviation
- visualize learning dynamics with uncertainty bands
- interpret high variance in reinforcement learning experiments

The main lesson from this stage is that reinforcement learning experiments should not rely on a single training run. Multi-seed evaluation provides a more honest and reliable view of agent performance.

---

## Future Research Direction

After implementing DQN, this project will move toward a more focused experimental question.

Two possible directions are:

1. How do experience replay buffer size and target network update frequency affect DQN training stability?
2. Can Double DQN reduce Q-value overestimation compared with vanilla DQN in small control environments?

The final research question will be selected after the DQN implementation is complete and initial CartPole results are available.

---

## Project Structure

```text
cartpole-to-dqn-reinforcement-learning/
│
├── README.md
├── notebooks/
│   ├── 00_dqn_preview.ipynb
│   ├── 00_python_basics.ipynb
│   ├── 00_numpy_matplotlib_basics.ipynb
│   └── 01_rl_basics.ipynb
│
├── results/
│   └── qlearning_cartpole_multiseed.png
│
├── src/
├── notes/
└── anaconda_projects/
```

---

## How to Run

Open the notebook:

```text
notebooks/01_rl_basics.ipynb
```

Then run the cells in order.

Required libraries:

```text
gymnasium
numpy
matplotlib
```

---

## Next Steps

The next stage is to implement **Deep Q-Networks (DQN)** on CartPole.

Planned additions:

- PyTorch-based Q-network
- Experience replay
- Target network
- DQN training loop
- Q-learning vs DQN comparison
- Multi-seed DQN learning curves

After DQN is working reliably on CartPole, the project will move to **LunarLander**, which will serve as the main environment for more rigorous algorithm comparisons.

Possible research-style experiments include:

- DQN vs Double DQN
- ablation study on replay buffer size
- ablation study on target network update frequency
- multi-seed performance comparison with mean ± standard deviation curves

Atari Pong is treated as an optional extension rather than the core goal, because Atari training is significantly more computationally expensive.

The long-term goal is to turn this repository from a basic implementation project into a small but well-documented reinforcement learning experimental pipeline.