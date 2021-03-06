# Snake-DQN

Created a gym environment of the Snake game and implemented a [Deep Q Network](https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf) to play in it.

To view the agent in Snake with a 9x9 grid, run [`view.py`](view.py)

https://user-images.githubusercontent.com/51470282/124981801-d42d1e00-dfea-11eb-8fb2-b2252aa4aec5.mp4

First run, training on 9x9 grid, around 3 hours with RTX 3060 Mobile:

<img src="misc/training1.png" width="500">

Second run, continuing the training on 9x9 grid, around 5 hours with RTX 3060 Mobile (this took more time than the first run even though the second run played less games because the games were longer, as the snake got better at staying alive):

<img src="misc/training2.png" width="500">

The grid dimensions can be changed by modifying the `DeepQNetwork` class' architecture in [`dqn.py`](dqn.py) and changing `grid_size` and making a new `model_path` with ".pt" ending in [`config.py`](config.py). Other constants in [`config.py`](config.py) can be changed, such as the learning rate or gamma. Then, to train, run [`train.py`](train.py), and then [`view.py`](view.py). Note that even when running [`view.py`](view.py), the deep q network has to have the same architecture as the corresponding ".pt" model.
