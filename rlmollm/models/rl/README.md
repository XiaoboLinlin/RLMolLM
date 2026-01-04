# Reinforcement Learning Module

This directory contains the reinforcement learning components for molecular generation optimization.

## Components

### PPOTrainer (`ppo_trainer.py`)
The main PPO (Proximal Policy Optimization) trainer class that handles:
- Molecule generation with probability tracking
- Reward calculation based on molecular properties
- PPO loss computation and model updates
- Value network integration

### MoleculeValueNetwork (`MoleculeValueNetwork.py`)
A neural network that estimates the value of molecule states for PPO training:
- Uses self-attention to process molecular sequences
- Provides value estimates for advantage computation
- Integrated with the PPO training loop

## Usage

The PPO trainer is automatically initialized and used by the Population class when `train_ppo()` is called:

```python
from population.population import Population

# Initialize population with your components
population = Population(gan_operators, scoring_operator, ...)

# Train using PPO
loss, avg_reward, valid_rate = population.train_ppo(
    dataloader=dataloader,
    ppo_epochs=4,
    clip_ratio=0.2,
    reward_scale=1.5,
    invalid_penalty=-0.9
)
```

## Architecture

The PPO implementation follows these key principles:
1. **Molecule Generation**: Uses masked language modeling to generate variations
2. **Reward Calculation**: Scores molecules using the provided scoring operator
3. **Policy Optimization**: Updates the generator using PPO objectives
4. **Value Learning**: Trains a value network to estimate molecule quality

## Integration

The RL module integrates seamlessly with the existing codebase:
- Uses existing GAN operators for generation
- Works with any scoring operator
- Supports scaffold-based generation
- Compatible with different masking modes 