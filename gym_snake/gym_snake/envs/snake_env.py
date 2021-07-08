import gym
from gym import error, spaces, utils
from gym.utils import seeding
import random
import pygame
import numpy as np

class SnakeEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self):
        self.gridSize = (9, 9)

        self.screen_is_on = False

        self.seed()
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=1, shape=(3, self.gridSize[0]*2-1, self.gridSize[1]*2-1), dtype=np.uint8)

        self.pos = [(2, self.gridSize[1]//2), (3, self.gridSize[1]//2), (4, self.gridSize[1]//2), (5, self.gridSize[1]//2)]
        self.dir = 'nope'
        self.oldDir = 'nope'

        self.food = (self.gridSize[0]-5, self.gridSize[1]//2)
        self.oldFood = (self.gridSize[0]-5, self.gridSize[1]//2)

        self.isDead = False
        self.growing = []
        self.stepsSinceEaten = 0
        self.maxSteps = self.gridSize[0] * self.gridSize[1] + 1
    
    def reset(self):
        self.pos = [(2, self.gridSize[1]//2), (3, self.gridSize[1]//2), (4, self.gridSize[1]//2), (5, self.gridSize[1]//2)]
        self.dir = 'nope'
        self.oldDir = 'nope'

        self.food = (self.gridSize[0]-5, self.gridSize[1]//2)
        self.oldFood = (self.gridSize[0]-5, self.gridSize[1]//2)

        self.isDead = False
        self.growing = []
        self.stepsSinceEaten = 0

        return self.state()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def move(self):
        if self.dir == 'nope' or self.isDead:
            return

        head = self.pos[-1]
        neck = self.pos[-2]
        if self.dir == 'r' and head[0] >= neck[0]:
            self.pos.append((head[0]+1, head[1]))

        elif self.dir == 'l' and head[0] <= neck[0]:
            self.pos.append((head[0]-1, head[1]))

        elif self.dir == 'u' and head[1] <= neck[1]:
            self.pos.append((head[0], head[1]-1))

        elif self.dir == 'd' and head[1] >= neck[1]:
            self.pos.append((head[0], head[1]+1))
        
        else:
            self.dir = self.oldDir
            self.move()
            return
        self.oldDir = self.dir
        self.pos.pop(0)

    def collision_detect(self):
        head = self.pos[-1]
        if head[0] < 0 or head[1] < 0 or head[0] >= self.gridSize[0] or head[1] >= self.gridSize[1] or head in self.pos[:-1]:
            return True
        else:
            return False

    def state(self):
        state = np.zeros((3, self.gridSize[1]*2-1, self.gridSize[0]*2-1), dtype=np.uint8)
        if self.isDead:
            return state
        state[1, self.food[1]*2, self.food[0]*2] = 1
        for idx, bodyPart in enumerate(self.pos):
            if idx == len(self.pos)-1:
                state[2, bodyPart[1]*2, bodyPart[0]*2] = 1
            else:
                state[0, bodyPart[1]*2, bodyPart[0]*2] = 1
                adjBodyPart = self.pos[idx+1]
                if bodyPart[1] == adjBodyPart[1]:
                    mid = bodyPart[0] + adjBodyPart[0]
                    state[0, bodyPart[1]*2, mid] = 1
                else:
                    mid = bodyPart[1] + adjBodyPart[1]
                    state[0, mid, bodyPart[0]*2] = 1
        return state

    def reward(self):
        reward = 0
        if self.pos[-1] == self.food:
            reward += 1
        elif self.isDead:
            reward -= 1
            
        return reward

    def step(self, action):
        if action == 'nope':
            return self.state(), self.reward(), self.isDead, {'episode': None, 'is_success': None, 'final_score': None}
        self.dir = ['u', 'r', 'd', 'l'][action]
        head = self.pos[-1]
        try:
            if self.growing[0] == self.oldFood:
                self.pos.insert(0, self.oldFood) 
                self.oldFood = self.food
                self.growing.pop(0)
        except:
            pass
        if head == self.food:
            if len(self.pos) == self.gridSize[0] * self.gridSize[1]:
                self.isDead = True
                return self.state(), self.reward(), self.isDead, {'episode': None, 'is_success': None, 'final_score': len(self.pos)}
            else:
                self.growing.append(head)
                self.stepsSinceEaten = 0
                while True:
                    self.food = (random.choice(range(self.gridSize[0])), random.choice(range(self.gridSize[1])))
                    if self.food not in self.pos:
                        break
        else:
            self.stepsSinceEaten += 1
        self.move()
        if self.collision_detect() or self.stepsSinceEaten >= self.maxSteps:
            self.isDead = True

        return self.state(), self.reward(), self.isDead, {'episode': None, 'is_success': None, 'final_score': len(self.pos) if self.isDead else None}

    def close(self):
        self.__init__()
        pygame.quit()

    def screen_on(self):
        pygame.init()
        self.squareSize = 50
        self.width = self.gridSize[0]*self.squareSize
        self.height = self.gridSize[1]*self.squareSize
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.fps = 19
        pygame.display.set_caption("Snake")

    # vv this took me ages >:( vv
    def render(self, mode='human', close=False):
        if not self.screen_is_on:
            self.screen_on()
            self.screen_is_on = True

        self.clock.tick(self.fps)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                self.isDead = True
                return

        self.screen.fill((51, 51, 51))
        pygame.draw.rect(self.screen, (200, 0, 0), ((self.food[0]+0.125)*self.squareSize, (self.food[1]+0.125)*self.squareSize, self.squareSize*0.75, self.squareSize*0.75))

        snakeThinness = 0.25
        for i in range(len(self.pos)):
            c, r = self.pos[i][0], self.pos[i][1]

            if i == 0:
                if c == self.pos[1][0]:
                    if r < self.pos[1][1]:
                        pygame.draw.rect(self.screen, (0, 200, 0), ((c+snakeThinness*0.5)*self.squareSize, (r+snakeThinness)*self.squareSize, (1-snakeThinness)*self.squareSize, (1-snakeThinness)*self.squareSize+1))
                    else:
                        pygame.draw.rect(self.screen, (0, 200, 0), ((c+snakeThinness*0.5)*self.squareSize, r*self.squareSize, (1-snakeThinness)*self.squareSize, (1-snakeThinness)*self.squareSize))
                else:
                    if c < self.pos[1][0]:
                        pygame.draw.rect(self.screen, (0, 200, 0), ((c+snakeThinness)*self.squareSize, (r+snakeThinness*0.5)*self.squareSize, (1-snakeThinness)*self.squareSize+1, (1-snakeThinness)*self.squareSize))
                    else:
                        pygame.draw.rect(self.screen, (0, 200, 0), (c*self.squareSize, (r+snakeThinness*0.5)*self.squareSize, (1-snakeThinness)*self.squareSize, (1-snakeThinness)*self.squareSize))

            elif i == len(self.pos)-1:
                if c == self.pos[-2][0]:
                    if r < self.pos[-2][1]:
                        pygame.draw.rect(self.screen, (0, 200, 0), ((c+snakeThinness*0.5)*self.squareSize, (r+snakeThinness)*self.squareSize, (1-snakeThinness)*self.squareSize, (1-snakeThinness)*self.squareSize+1))
                    else:
                        pygame.draw.rect(self.screen, (0, 200, 0), ((c+snakeThinness*0.5)*self.squareSize, r*self.squareSize, (1-snakeThinness)*self.squareSize, (1-snakeThinness)*self.squareSize))
                else:
                    if c < self.pos[-2][0]:
                        pygame.draw.rect(self.screen, (0, 200, 0), ((c+snakeThinness)*self.squareSize, (r+snakeThinness*0.5)*self.squareSize, (1-snakeThinness)*self.squareSize+1, (1-snakeThinness)*self.squareSize))
                    else:
                        pygame.draw.rect(self.screen, (0, 200, 0), (c*self.squareSize, (r+snakeThinness*0.5)*self.squareSize, (1-snakeThinness)*self.squareSize, (1-snakeThinness)*self.squareSize))

            else:
                if self.pos[i-1][1] == r == self.pos[i+1][1]:
                    pygame.draw.rect(self.screen, (0, 200, 0), (c*self.squareSize, (r+snakeThinness*0.5)*self.squareSize, self.squareSize, (1-snakeThinness)*self.squareSize))
                elif self.pos[i-1][0] == c == self.pos[i+1][0]:
                    pygame.draw.rect(self.screen, (0, 200, 0), ((c+snakeThinness*0.5)*self.squareSize, r*self.squareSize, (1-snakeThinness)*self.squareSize, self.squareSize))
                else:
                    if (self.pos[i-1][0] < c and self.pos[i+1][1] < r) or (self.pos[i-1][1] < r and self.pos[i+1][0] < c):
                        pygame.draw.rect(self.screen, (0, 200, 0), (c*self.squareSize, (r+snakeThinness*0.5)*self.squareSize, (1-snakeThinness*0.5)*self.squareSize-1, (1-snakeThinness)*self.squareSize))
                        pygame.draw.rect(self.screen, (0, 200, 0), ((c+snakeThinness*0.5)*self.squareSize, r*self.squareSize, (1-snakeThinness)*self.squareSize, (1-snakeThinness*0.5)*self.squareSize-1))
                    elif (self.pos[i-1][0] < c and self.pos[i+1][1] > r) or (self.pos[i-1][1] > r and self.pos[i+1][0] < c):
                        pygame.draw.rect(self.screen, (0, 200, 0), (c*self.squareSize, (r+snakeThinness*0.5)*self.squareSize, (1-snakeThinness*0.5)*self.squareSize-1, (1-snakeThinness)*self.squareSize))
                        pygame.draw.rect(self.screen, (0, 200, 0), ((c+snakeThinness*0.5)*self.squareSize, (r+snakeThinness*0.5)*self.squareSize, (1-snakeThinness)*self.squareSize, (1-snakeThinness*0.5)*self.squareSize+1))
                    elif (self.pos[i-1][1] > r and self.pos[i+1][0] > c) or (self.pos[i-1][0] > c and self.pos[i+1][1] > r):
                        pygame.draw.rect(self.screen, (0, 200, 0), ((c+snakeThinness*0.5)*self.squareSize, (r+snakeThinness*0.5)*self.squareSize, (1-snakeThinness*0.5)*self.squareSize+1, (1-snakeThinness)*self.squareSize))
                        pygame.draw.rect(self.screen, (0, 200, 0), ((c+snakeThinness*0.5)*self.squareSize, (r+snakeThinness*0.5)*self.squareSize, (1-snakeThinness)*self.squareSize, (1-snakeThinness*0.5)*self.squareSize+1))
                    else:
                        pygame.draw.rect(self.screen, (0, 200, 0), ((c+snakeThinness*0.5)*self.squareSize, (r+snakeThinness*0.5)*self.squareSize, (1-snakeThinness*0.5)*self.squareSize+1, (1-snakeThinness)*self.squareSize))
                        pygame.draw.rect(self.screen, (0, 200, 0), ((c+snakeThinness*0.5)*self.squareSize, r*self.squareSize, (1-snakeThinness)*self.squareSize, (1-snakeThinness*0.5)*self.squareSize-1))
            
        pygame.display.flip()
    
    def showScore(self):
        font = pygame.font.Font('freesansbold.ttf', 32)
        text = font.render(f'Final score: {len(self.pos)-4}', True, (0, 255, 0), (0, 0, 255))
        textRect = text.get_rect()
        textRect.center = (self.width // 2, self.height // 2)

        while True:
            self.screen.fill((204, 204, 204))
            self.screen.blit(text, textRect)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                pygame.display.update()