import numpy as np
import utils
import random
import math


class Agent:

    def __init__(self, actions, Ne, C, gamma):
        self.actions = actions
        self.Ne = Ne # used in exploration function
        self.C = C
        self.gamma = gamma

        # Create the Q and N Table to work with
        self.Q = utils.create_q_table()
        self.N = utils.create_q_table()
        self.reset()

    def train(self):
        self._train = True

    def eval(self):
        self._train = False

    # At the end of training save the trained model
    def save_model(self,model_path):
        utils.save(model_path, self.Q)

    # Load the trained model for evaluation
    def load_model(self,model_path):
        self.Q = utils.load(model_path)

    def reset(self):
        self.points = 0
        self.s = None
        self.a = None

    def act(self, state, points, dead):
        '''
        :param state: a list of [snake_head_x, snake_head_y, snake_body, food_x, food_y] from environment.
        :param points: float, the current points from environment
        :param dead: boolean, if the snake is dead
        :return: the index of action. 0,1,2,3 indicates up,down,left,right separately

        TODO: write your function here.
        Return the index of action the snake needs to take, according to the state and points known from environment.
        Tips: you need to discretize the state to the state space defined on the webpage first.
        (Note that [adjoining_wall_x=0, adjoining_wall_y=0] is also the case when snake runs out of the 480x480 board)

        '''
        state_next = self.discretizeState(self.convertToGrid(state))
        # print(points)
        # update Q Table
        if self._train and self.s != None:
            if dead:
                reward = -1
            elif points > self.points:
                reward = 1
            else:
                reward = -0.1
            maxQ_next = -math.inf
            for a in range(4):
                maxQ_next = max(maxQ_next, self.Q[state_next + (a,)])
            self.Q[self.s + (self.a,)] += self.C / (self.C + self.N[self.s + (self.a,)]) * (reward + self.gamma * maxQ_next - self.Q[self.s + (self.a,)])

        # choose action
        if not dead:
            best_f = -math.inf
            best_a = 0
            for a in range(4):
                if self.N[state_next + (a, )] < self.Ne:
                    curr_f = 1
                else:
                    curr_f = self.Q[state_next + (a,)]
                if curr_f >= best_f:
                    best_f = curr_f
                    best_a = a
            self.a = best_a
            self.s = state_next
            self.points = points
            self.N[self.s + (self.a,)] += 1
        else:
            self.reset()
        return self.a

    def convertToGrid(self, state):
        snake_head_x = state[0] // utils.GRID_SIZE
        snake_head_y = state[1] // utils.GRID_SIZE
        food_x = state[3] // utils.GRID_SIZE
        food_y = state[4] // utils.GRID_SIZE
        snake_body = []
        for body in state[2]:
            body_x, body_y = body[0] // utils.GRID_SIZE, body[1] // utils.GRID_SIZE
            snake_body.append((body_x, body_y))
        return [snake_head_x, snake_head_y, snake_body, food_x, food_y]

    def discretizeState(self, state):
        if state[0] == 1:
            adjoining_wall_x = 1
        elif state[0] == 12:
            adjoining_wall_x = 2
        else:
            adjoining_wall_x = 0

        if state[1] == 1:
            adjoining_wall_y = 1
        elif state[1] == 12:
            adjoining_wall_y = 2
        else:
            adjoining_wall_y = 0

        if state[0] <= 0 or state[0] >= 13 or state[1] <= 0 or state[0] >=13:
            adjoining_wall_x, adjoining_wall_y = 0, 0

        if state[0] > state[3]:
            food_dir_x = 1
        elif state[0] < state[3]:
            food_dir_x = 2
        else:
            food_dir_x = 0

        if state[1] > state[4]:
            food_dir_y = 1
        elif state[1] < state[4]:
            food_dir_y = 2
        else:
            food_dir_y = 0

        if (state[0], state[1] - 1) in state[2]:
            adjoining_body_top = 1
        else:
            adjoining_body_top = 0

        if (state[0], state[1] + 1) in state[2]:
            adjoining_body_bottom = 1
        else:
            adjoining_body_bottom = 0

        if (state[0] - 1, state[1]) in state[2]:
            adjoining_body_left = 1
        else:
            adjoining_body_left = 0

        if (state[0] + 1, state[1]) in state[2]:
            adjoining_body_right = 1
        else:
            adjoining_body_right = 0

        return (adjoining_wall_x, adjoining_wall_y, food_dir_x, food_dir_y, adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right)
