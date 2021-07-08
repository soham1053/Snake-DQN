import gym
import gym_snake
import pygame
import matplotlib.pyplot as plt
import numpy as np

env = gym.make('snake-v0')
env.reset()
env.render()
env.fps = 8
action = 'nope'
done = False
while not done:
    keys = pygame.key.get_pressed()
    if keys[pygame.K_UP]:
        action = 0
    elif keys[pygame.K_RIGHT]:
        action = 1
    elif keys[pygame.K_DOWN]:
        action = 2
    elif keys[pygame.K_LEFT]:
        action = 3
    state, _, done, _ = env.step(action)
    env.render()

env.showScore()