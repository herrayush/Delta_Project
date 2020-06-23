import tkinter
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import time
import sys
import tkinter as tk
import pandas as pd




pixel_count = 40  
grid_height= int(input("Enter the number of rows:  ")) 
grid_width= int(input("Enter the number of cols:  ")) 
num_holes = int(input("Enter the number of holes:  "))
epoch_count = int(input("Enter the nuber of epochs: "))




class rl_maze():
    def __init__(self):
        self.maze_screen = tk.Tk()
        self.maze_screen.title('RL maze')
        self.maze_screen.geometry('{0}x{1}'.format(grid_width* pixel_count, grid_height* pixel_count))
        self.action_space = ['u', 'd', 'l', 'r'] 
        self.n_actions = len(self.action_space)
        self.build_grid()
        
        
    def build_grid(self):
        self.hole_list = []
        self.canvas = tk.Canvas(self.maze_screen, bg='#fffdd0',height=grid_height* pixel_count,width=grid_width* pixel_count)

        for c in range(0, grid_width* pixel_count, pixel_count):
            x0, y0, x1, y1 = c, 0, c, grid_width* pixel_count
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, grid_height* pixel_count, pixel_count):
            x0, y0, x1, y1 = 0, r, grid_height* pixel_count, r
            self.canvas.create_line(x0, y0, x1, y1)
        origin = np.array([20, 20])

        for i in range(num_holes):
            a,b = map(int, input("Enter the coordinates of the {}th hole: ".format(i+1)).split())
            temp_hole_center = origin + np.array([pixel_count * a, pixel_count*b])
            self.temp_hole = self.canvas.create_rectangle(
                    temp_hole_center[0] - 15, temp_hole_center[1] - 15,
                    temp_hole_center[0] + 15, temp_hole_center[1] + 15,
                    fill='black')
            self.hole_list.append(self.canvas.coords(self.temp_hole))

        
        a,b = map(int, input("Enter the co-ordinates of the goal: ").split())
        oval_center = origin + np.array([pixel_count * a, pixel_count*b])
        self.oval = self.canvas.create_oval(
            oval_center[0] - 15, oval_center[1] - 15,
            oval_center[0] + 15, oval_center[1] + 15,
            fill='green')

        self.start = self.canvas.create_oval(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='yellow')
        self.canvas.pack()

    def render(self):
        time.sleep(0.1)
        self.maze_screen.update()

    def reset(self):
        self.maze_screen.update()
        time.sleep(0.5)
        self.canvas.delete(self.start)
        origin = np.array([20, 20])
        self.start = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='yellow')
        return self.canvas.coords(self.start)


    def get_state_reward(self, action):
        tracker= self.canvas.coords(self.start)
        action_list= np.array([0, 0])
        if action == 0:  
            if tracker[1] > pixel_count:
                action_list[1] -= pixel_count
        elif action == 1:
            if tracker[1] < (grid_height- 1) * pixel_count:
                action_list[1] += pixel_count
        elif action == 2:
            if tracker[0] < (grid_width- 1) * pixel_count:
                action_list[0] += pixel_count
        elif action == 3:
            if tracker[0] > pixel_count:
                action_list[0] -= pixel_count

        self.canvas.move(self.start, action_list[0], action_list[1])

        s_ = self.canvas.coords(self.start)

        if s_ == self.canvas.coords(self.oval):
            reward = 1
            done = True
            s_ = 'terminal'
        elif s_ in self.hole_list:
            reward = -1
            done = True
            s_ = 'terminal'
        else:
            reward = 0
            done = False

        return s_, reward, done





class rl_table:

    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.1):
        self.actions = actions  
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, observation):
        self.add_state(observation)
        if np.random.uniform() < self.epsilon:
            action = np.random.choice(self.actions)

        else:
            state_action = self.q_table.loc[observation, :]
            state_action = state_action.reindex(np.random.permutation(state_action.index)) # some actions have same value
            action = state_action.idxmax()
        return action

    def learn(self, s, a, r, s_):

        self.add_state(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()
        else:
            q_target = r
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)

    def add_state(self, state):
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(pd.Series([0]*len(self.actions),index=self.q_table.columns,name=state))



epochs = range(epoch_count)
movements = []
rewards = []


def run_experiment():
    print("Training Started!!!")
    for epoch in epochs:
        print("Epoch %s/%s." %(epoch+1, epoch_count))
        observation = environment.reset()
        moves = 0

        while True:
            environment.render()
            action = rl_bot.choose_action(str(observation))
            observation_, reward, done = environment.get_state_reward(action)
            moves +=1
            rl_bot.learn(str(observation), action, reward, str(observation_))
            observation = observation_
            if done:
                movements.append(moves) 
                rewards.append(reward)
                print("Reward: {0}, Moves: {1}".format(reward, moves))
                break
            
    print('Training Done!')
    final_figure()


def final_figure():
    plt.figure()
    plt.step(epochs, rewards)
    plt.xlabel("Epoch")
    plt.ylabel("Reward")
    plt.show()


if __name__ == "__main__":
    environment = rl_maze()
    rl_bot = rl_table(actions=list(range(environment.n_actions)))
    environment.maze_screen.after(10, run_experiment)
    environment.maze_screen.mainloop()
