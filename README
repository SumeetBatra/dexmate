## Code Structure
- 'custom_robots/vega.py' contains code for loading the URDF, defining the controllers, and setting up the vega robot in the simulation.
- `envs/vega_pick_cube.py` contains the custom environment for grasping the cube from randomized positions
- `scripts/ppo.py` is the script to train the vega robot to pick up the cube using PPO.
using the vega robot.

## Installation

1. Create a conda env
```bash
conda create -n dexmate python=3.11
conda activate dexmate
```

2. Install dependencies from requirements.txt
```bash
pip install -r requirements.txt
```

3. Run PPO from root directory with wandb logging support

```python
python -m scripts.ppo --env_id=PickCubeVega --num_envs=2048 --update_epochs=8 --num_minibatches=32 --total_timesteps=100_000_000
--eval_freq=10 --num-steps=20 --control-mode=pd_joint_pos --track
```

