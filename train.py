import gym
from dqn import Agent
import utils
import numpy as np
from config import *
import gym_snake
import time

if __name__ == '__main__':
    env = gym.make(env_name)
    env.gridSize = grid_size
    agent = Agent(input_dims=env.observation_space.shape, n_actions=env.action_space.n)
    scores, eps_history = [], []

    env.render()
    env.fps = 0

    chooseActionTime, stepTime, storeTransitionTime, learnTime = 0, 0, 0, 0
    totalTime = time.time()
    for i in range(1, n_games+1):
        score = 0
        done = False
        observation = env.reset()
        env.render()
        while not done:
            s = time.time()
            action = agent.choose_action(observation)
            chooseActionTime += time.time() - s
            s = time.time()
            observation_, reward, done, info = env.step(action)
            stepTime += time.time() - s
            score += reward
            env.render()
            s = time.time()
            agent.store_transition(observation, action, reward, 
                                    observation_, done)
            storeTransitionTime += time.time() - s
            s = time.time()
            agent.learn()
            learnTime += time.time() - s
            observation = observation_
        scores.append(score)
        eps_history.append(agent.epsilon)
        avg_score = np.mean(scores[-100:])
        if i % 100 == 0:
            print('episode ', i, 'score %.2f' % score, 
                    'average score %.2f' % avg_score, 
                    'epsilon %.2f' % agent.epsilon, 
                    f'chooseAction %.2f' % chooseActionTime, 
                    f'step %.2f' % stepTime, 
                    f'storeTransition %.2f' % storeTransitionTime, 
                    f'learn %.2f' % learnTime, 
                    f'total %.2f' % (time.time() - totalTime), end='\n\n')
            chooseActionTime, stepTime, storeTransitionTime, learnTime = 0, 0, 0, 0
            totalTime = time.time()
        if i % 500 == 0:
            agent.save(model_path)
    env.close()

    agent.save(model_path)
    
    x = [i+1 for i in range(n_games)]
    utils.plotLearning(x, scores, eps_history)