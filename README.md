# DeepEON

**Deep Reinforcement Learning Applied in Elastic Optical Networks Using Game Representation**

A deep reinforcement learning framework that trains DQN agents to solve the Routing and Spectrum Allocation (RSA) problem in Elastic Optical Networks (EONs) using an arcade-game-based visual representation.

> 4th Year Project -- Department of Electronic & Electrical Engineering, UCL
>
> Authors: Moshe Simon, Rafael Asensio, Andrey Staniukynas, Alina Stanoiu, David Sal
>
> Supervisor: Dr Alejandra Beghelli
>
> April 2023

## Overview

Elastic Optical Networks divide the spectrum into flexible 12.5 GHz Frequency Slot Units (FSUs), enabling more efficient bandwidth allocation compared to fixed-grid WDM networks. The RSA problem -- finding a feasible route and spectrum allocation for traffic demands while satisfying **continuity** (same slots along entire route) and **contiguity** (adjacent slots on each link) constraints -- is NP-hard and traditionally solved with heuristics like KSP-FF (K-Shortest Path with First-Fit).

DeepEON represents the RSA problem as an arcade game where a DQN agent learns to allocate spectrum slots on a 6-node network topology by interacting with a visual grid rendered via Pygame. The agent observes the grid as an RGB image and is trained using a CNN policy from Stable Baselines 3.

### Two Approaches

1. **RSA Game** (Games 6-8): The agent handles both routing and spectrum allocation jointly. Multiple game versions were developed with progressive improvements including a centralised config file, new environments, episode-ending mechanisms, and network visualisation.

2. **Routing Game** (Games 3-4): The agent focuses solely on routing, while a First-Fit heuristic handles spectrum allocation. This decomposition showed the best performance, coming within ~10% of the KSP-FF baseline.

## Project Structure

```
DeepEON/
├── ArcadeGame/
│   ├── Games/                  # Game environment implementations
│   │   ├── game6.py            # Graph-based RSA game
│   │   └── game7.py            # RSA game with network visualisation
│   ├── envs/                   # OpenAI Gym environment wrappers
│   │   ├── custom_env.py       # Env 1: Arrow-based action space (LEFT/RIGHT/ENTER)
│   │   ├── custom_env2.py      # Env 2: Teleport action space (direct slot selection)
│   │   └── custom_env3.py      # Env 3: Node-based pathfinding with grid rendering
│   ├── callbacks/
│   │   └── LogGifsCallback.py  # W&B callback for logging episode GIFs
│   ├── config.py               # Centralised configuration (all hyperparameters)
│   ├── train.py                # Main DQN training script
│   ├── evaluate.py             # Deterministic evaluation over N episodes
│   ├── record.py               # Record agent gameplay to MP4
│   ├── play.py                 # Human-playable game interface
│   ├── heurisic.py             # KSP-FF heuristic baseline
│   ├── plot.py                 # Result visualisation
│   └── example.py              # Experimental DreamerV3 integration
├── Models/                     # Saved model checkpoints (.zip)
├── Evaluations/                # Evaluation results (JSON)
├── Recordings/                 # Agent gameplay videos (MP4)
├── tensorboardEON/             # TensorBoard logs
├── requirements.txt            # Python dependencies
├── environment.yml             # Conda environment specification
└── sweep.yaml                  # W&B hyperparameter sweep configuration
```

## Environments

| Environment | Action Space | Description |
|---|---|---|
| **Env 1** (Arrows) | 3 discrete (LEFT, RIGHT, ENTER) | Selection window moved incrementally; reward only on ENTER |
| **Env 2** (Teleport) | `NUM_SLOTS * K + K - 1` discrete | Direct slot position selection; immediate reward per action |
| **Env 3** (Node Pathfinding) | Grid-based discrete | Agent selects nodes to build paths; visual grid with colour-coded slots |

Env 2 outperformed Env 1 by **37.1%** due to reduced sparse reward effects -- each action maps directly to a slot, providing immediate feedback.

## Setup

### Prerequisites

- Python 3.8
- Conda (recommended)

### Installation

```bash
# Create and activate conda environment
conda env create -f environment.yml
conda activate eon

# Or install with pip
pip install -r requirements.txt
```

### Key Dependencies

- `stable-baselines3` -- DQN agent implementation
- `pytorch 1.10.2` -- Neural network backend
- `pygame` / `arcade 2.6.10` -- Game rendering
- `gym 0.19.0` -- Environment interface
- `wandb` -- Experiment tracking
- `networkx` -- Graph/network topology

## Usage

### Configuration

All parameters are controlled through `ArcadeGame/config.py`:

- **Game parameters**: game number, window size, colours
- **Environment parameters**: reward values, number of slots, number of routes (K), episode-ending mechanism
- **Agent hyperparameters**: batch size, buffer size, learning rate, gamma, exploration schedule
- **Training parameters**: total timesteps, seed

### Training

```bash
python ArcadeGame/train.py
```

Models are saved to `Models/{naming_convention}/model.zip` and metrics are logged to both TensorBoard and Weights & Biases.

### Hyperparameter Sweep

```bash
wandb sweep sweep.yaml
wandb agent <sweep_id>
```

Uses Bayesian optimisation to search over: `gamma`, `learning_rate`, `batch_size`, `buffer_size`, `exploration_final_eps`, `exploration_fraction`, `target_update_interval`, `learning_starts`, `train_freq`.

### Evaluation

```bash
python ArcadeGame/evaluate.py
```

Runs 1000 episodes with deterministic policy and saves results to `Evaluations/`.

### Recording

```bash
python ArcadeGame/record.py
```

Records agent gameplay as MP4 videos in `Recordings/`.

### Play Manually

```bash
python ArcadeGame/play.py
```

Play the game yourself to understand the environment dynamics.

## Model Architecture

- **Algorithm**: DQN (Deep Q-Network) with CNN policy
- **Observation**: RGB image of the game grid (slot allocation state)
- **Network**: Convolutional neural network processes pixel-based observations

| Hyperparameter | Default Value |
|---|---|
| Total timesteps | 5,000,000 |
| Buffer size | 100,000 |
| Batch size | 64 |
| Learning rate | 0.0001 |
| Gamma | 0.1 |
| Exploration fraction | 0.8 |
| Final epsilon | 0.1 |
| Target update interval | 20,000 |
| Learning starts | 100,000 |

## Results

### RSA Game

- **Env 2 (Teleport)** outperformed **Env 1 (Arrows)** by 37.1%
- Removing routing (K=1) reduced the heuristic's advantage from 78% to 47.5%
- The agent did not surpass the KSP-FF heuristic in any RSA configuration
- DreamerV3 world model was also tested but did not improve over DQN

### Routing Game

- The routing-only game with FF spectrum allocation showed the most promising results
- After hyperparameter optimisation, the agent came within **10.2%** of the KSP-FF heuristic
- The agent achieved higher network utility (36.3% vs 34.2%) but used longer routes (~40% more links), leading to lower overall reward
- Decomposing RSA into routing + spectrum allocation sub-problems improves performance

## Naming Convention

Model files follow the pattern:

```
{slots}_{screen_slots}_{K}_{solution_reward}_{rejection_reward}_{gap_rejection}_{seed}_{env}_{episode_end}_{end_limit}
```

Example: `8_8_3_10_-50_-5_0_3_2_9999` = 8 slots, K=3 routes, +10 solution reward, -50 rejection, Env 3, episode end mechanism 2, end limit 9999.

## Experiment Tracking

Training runs are logged to [Weights & Biases](https://wandb.ai) under the project `DeepEON`, including:

- Episode reward curves
- Loss and Q-value metrics
- GIF snapshots of agent gameplay every 100K episodes
- Hyperparameter sweep results

TensorBoard logs are stored in `tensorboardEON/` and can be viewed with:

```bash
tensorboard --logdir tensorboardEON/
```

## License

This project was developed as part of the UCL EEE 4th Year Project (2022-2023).
