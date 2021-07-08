# Snake-DQN
---

Created a gym environment of the Snake game and implemented a [Deep Q Network](https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf) to play in it.

To view the agent in Snake 9x9 grid, run [`view.py`](view.py)




The grid dimensions can be changed by modifying the `DeepQNetwork` class' architecture in [`dqn.py`](dqn.py) and changing `grid_size` and making a new `model_path` with ".pt" ending in [`config.py`](config.py). Other constants in [`config.py`](config.py) can be changed, such as the learning rate or gamma. Then, to train, run [`train.py`](train.py), and then [`view.py`](view.py). Note that even when running [`view.py`](view.py), the deep q network has to have the same architecture as the corresponding ".pt" model.