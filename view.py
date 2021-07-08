import gym
import gym_snake
from dqn import Agent
import utils
import numpy as np
from config import *

if __name__ == '__main__':
    env = gym.make(env_name)
    env.gridSize = grid_size
    agent = Agent(input_dims=env.observation_space.shape, n_actions=env.action_space.n, model_path=model_path, 
                    epsilon=0, eps_end=0)
    scores = []
    n_games = 10

    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()
        env.render()
        while not done:
            action = agent.choose_action(observation)
            observation, reward, done, info = env.step(action)
            env.render()
        score = info['final_score']
        scores.append(score)

        print('episode', i, 'score', score)
    
    utils.plotScores(scores)