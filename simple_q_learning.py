import numpy as np
import cv2

# Constants
LOAD_AND_SAVE_LEARNED_MODEL = False
SAVE_FILENAME = 'learned_grid.npy'
GUI_REFRESH_RATE = 10 #ms
GUI_UPDATE_STEP = 5000
SHOW_STEP = 500
# Grid definitions
GRID_SIZE = (10,10)
GRID_TERMINALS = [(3,3),(7,5)]
GRID_OBSTACLES = [(2,2),(2,3),(2,8),(1,7)]
GOAL = [(8,8)]
# GUI values
GUI_DIMENSION = (1000,1000)
ARROW_REDUCTION = 20
# Hyper parameters
ACTIONS = ['up','down','left','right']
EPSILON = 0.7
GAMMA = 0.95
ALPHA = 0.1
N_EPISODES = 500000

# The class providing a graphical interface
class GUI():
    def __init__(self, dimensions, grid, goal = [], terminals = [], obstacles = [], mouse_detect = False):
        self.grid_shape = grid
        self.dimensions = dimensions
        self.display = np.ones((self.dimensions[0],self.dimensions[1],3),dtype=np.uint8)*255
        self.display_buffer = None
        self.shape = self.display.shape
        self.grid_element_size = (self.shape[0]/self.grid_shape[0], self.shape[1]/self.grid_shape[1])
        self.line_colour = (255,127,0)
        self.goal_colour = (0,250,0)
        self.terminal_colour = (0,0,255)
        self.obstacle_colour = (0,0,0)
        self.player_colour = (0,127,255)
        self.arrow_colour = (180,180,180)
        self.episode_colour = (127,127,180)
        self.draw_grid(grid)
        self.draw_goal(goal)
        self.draw_obstacles(obstacles)
        self.draw_terminals(terminals)
        self.clicked_points = []
        self.user_input = None
        cv2.namedWindow('Environment')
        cv2.imshow('Environment',self.display)
        if mouse_detect is True:
            cv2.setMouseCallback('Environment',self.mouse_clicked)
            self.ix,self.iy = None,None

    def draw_grid(self, grid):
        shift_position = v_shift = self.shape[0]/grid[0]
        while(shift_position < self.shape[0]):
            cv2.line(self.display,(shift_position,0),(shift_position,self.shape[1]),self.line_colour,thickness=5)
            shift_position += v_shift
        shift_position = h_shift = self.shape[1]/grid[1]
        while(shift_position < self.shape[1]):
            cv2.line(self.display,(0,shift_position),(self.shape[0],shift_position),self.line_colour,thickness=5)
            shift_position += h_shift

    def draw_goal(self, goals):
        for goal in goals:
            x1,y1 = goal[0]*self.grid_element_size[0],goal[1]*self.grid_element_size[1]
            x2,y2 = x1+self.grid_element_size[0],y1+self.grid_element_size[1]
            cv2.rectangle(self.display,(x1,y1),(x2,y2),self.goal_colour,thickness=cv2.FILLED)

    def draw_obstacles(self, obstacles):
        for obs in obstacles:
            x1,y1 = obs[0]*self.grid_element_size[0],obs[1]*self.grid_element_size[1]
            x2,y2 = x1+self.grid_element_size[0],y1+self.grid_element_size[1]
            cv2.rectangle(self.display,(x1,y1),(x2,y2),self.obstacle_colour,thickness=cv2.FILLED)

    def draw_terminals(self, terminals):
        for term in terminals:
            x1,y1 = term[0]*self.grid_element_size[0],term[1]*self.grid_element_size[1]
            x2,y2 = x1+self.grid_element_size[0],y1+self.grid_element_size[1]
            cv2.rectangle(self.display,(x1,y1),(x2,y2),self.terminal_colour,thickness=cv2.FILLED)

    def update(self,episode_number, player_position, q_table=None, actions=None):
        self.display_buffer = np.copy(self.display)
        r = min(self.grid_element_size)/2
        x,y = (player_position[0]*self.grid_element_size[0])+(self.grid_element_size[0]/2),(player_position[1]*self.grid_element_size[1])+(self.grid_element_size[1]/2)
        cv2.circle(self.display_buffer,(x,y),r,self.player_colour,thickness=cv2.FILLED)
        if q_table is not None:
            for i in range(self.grid_shape[0]):
                for j in range(self.grid_shape[1]):
                    x1,y1 = (i*self.grid_element_size[0]),(j*self.grid_element_size[1])
                    x2,y2 = x1+self.grid_element_size[0],y1+self.grid_element_size[1]
                    if not np.any(q_table[i,j]):
                        r = min(self.grid_element_size)/8
                        x,y = (x1+x2)/2,(y1+y2)/2
                        cv2.circle(self.display_buffer,(x,y),r,self.arrow_colour,thickness=3)
                    elif np.unique(q_table[i,j]).size==1:
                        x,y = (x1+x2)/2,(y1+y2)/2
                        cv2.line(self.display_buffer,(x-10,y),(x+10,y),self.arrow_colour)
                    else:
                        if(actions[np.argmax(q_table[(i,j)])] == 'up'):
                            x1,y1,x2,y2 = (x1+x2)/2,y2-ARROW_REDUCTION,(x1+x2)/2,y1+ARROW_REDUCTION
                        elif(actions[np.argmax(q_table[(i,j)])] == 'down'):
                            x1,y1,x2,y2 = (x1+x2)/2,y1+ARROW_REDUCTION,(x1+x2)/2,y2-ARROW_REDUCTION
                        elif(actions[np.argmax(q_table[(i,j)])] == 'left'):
                            x1,y1,x2,y2 = x2-ARROW_REDUCTION,(y1+y2)/2,x1+ARROW_REDUCTION,(y1+y2)/2
                        elif(actions[np.argmax(q_table[(i,j)])] == 'right'):
                            x1,y1,x2,y2 = x1+ARROW_REDUCTION,(y1+y2)/2,x2-ARROW_REDUCTION,(y1+y2)/2
                        cv2.arrowedLine(self.display_buffer,(x1,y1),(x2,y2),self.arrow_colour,thickness=4)
        cv2.putText(self.display_buffer,'Episode: '+str(episode_number),(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,self.episode_colour,thickness=4)
        cv2.imshow('Environment',self.display_buffer)
        self.user_input = cv2.waitKey(GUI_REFRESH_RATE) & 0xFF

    def reset_frame(self, goal, obstacles, terminals):
        self.display = np.ones((self.dimensions[0],self.dimensions[1],3),dtype=np.uint8)*255
        self.draw_grid(self.grid_shape)
        self.draw_goal(goal)
        self.draw_obstacles(obstacles)
        self.draw_terminals(terminals)

    def mouse_clicked(self,event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            x,y = int(x/self.grid_element_size[0]),int(y/self.grid_element_size[1])
            self.clicked_points.append((x,y))

    def check_mouse(self):
        c_p = []
        if (len(self.clicked_points) is not 0):
            c_p = self.clicked_points
            self.clicked_points = []
        return c_p

    def check_input(self):
        if self.user_input != 255:
            return self.user_input

# The class defining the environment
class Environment():
    def __init__(self, gui, grid, goal, origin = (0,0), terminals = [], obstacles = []):
        self.gui = gui
        self.agent_state = origin
        self.origin = origin
        self.goal = goal
        self.terminals = terminals
        self.obstacles = obstacles
        self.grid = grid

    def take_action(self, action):
        agent_state_buffer = list(self.agent_state)
        # Decide next state
        if action == 'up':
            agent_state_buffer[1] -= 1
        elif action == 'down':
            agent_state_buffer[1] += 1
        elif action == 'right':
            agent_state_buffer[0] += 1
        elif action == 'left':
            agent_state_buffer[0] -= 1
        if (agent_state_buffer[0] >= self.grid[0]) or (agent_state_buffer[0] < 0) or (agent_state_buffer[1] >= self.grid[1]) or (agent_state_buffer[1] < 0) or (tuple(agent_state_buffer) in self.obstacles):
            agent_state_buffer =  self.agent_state
        self.agent_state = tuple(agent_state_buffer)
        # Decide reward and check if next state is terminal
        terminating = False
        reward = 0
        if self.agent_state in self.terminals:
            reward = -1
            terminating=True
        elif self.agent_state in self.goal:
            reward = 1
            terminating=True
        else:
            reward = 0
        return self.agent_state, reward, terminating

    def get_state(self):
        return self.agent_state

    def reset(self):
        self.agent_state = self.origin

    def redefine(self, goal = None, origin = None, terminals = None, obstacles = None):
        if goal is not None:
            self.goal = goal
        if origin is not None:
            self.origin = origin
        if terminals is not None:
            self.terminals = terminals
        if obstacles is not None:
            self.obstacles = obstacles

# The class defining the Agent
class Agent():
    def __init__(self, grid, actions, epsilon, gamma, alpha, load_learned = False):
        self.q_table = np.zeros(grid+(len(actions),),dtype=float)
        #self.q_table = np.ones(grid+(len(actions),),dtype=float)*0.0001
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.grid = grid # later change to auto add a new state if found, dynamic env mapping
        self.actions = actions
        if load_learned == True:
            try:
                self.load_learned_model(SAVE_FILENAME)
            except:
                print 'No saved model found. Resetting all values.'

    def pick_action(self, state):
        action = None
        random_probability = np.random.rand()
        if(random_probability>self.epsilon) or (not np.any(self.q_table[state])): # if Exploration
            action = self.actions[np.random.randint(low=0,high=len(self.actions))] # Select random action
        else: # Act greedy
            action = self.actions[np.argmax(self.q_table[state])] # Select best action based on value
        return action

    def take_step(self, env, action):
        new_state,reward,is_terminal = env.take_action(action)
        return new_state,reward,is_terminal

    def update_q_table(self,s,a,n_s,r,is_terminal):
        if is_terminal==True:
            self.q_table[s+(self.actions.index(a),)] += self.alpha*(r - self.q_table[s+(self.actions.index(a),)])
        else:
            self.q_table[s+(self.actions.index(a),)] += self.alpha*(r + self.gamma*np.amax(self.q_table[n_s]) - self.q_table[s+(self.actions.index(a),)])

    def load_learned_model(self, filename):
        loaded_q_table = np.load(filename)
        if loaded_q_table.shape == self.q_table.shape:
            self.q_table = loaded_q_table
            print 'Learned values loaded'

    def save_learned_model(self, filename):
        np.save(filename,self.q_table)
        #print 'Learned values saved'

# Run a episode
def run_episode(episode_number, agnt, env, gui=None):
    user_input = None
    env.reset()
    if gui is not None:
        gui.update(episode_number,env.agent_state,agnt.q_table,agnt.actions)
    while(True):
        s = env.get_state()
        a = agnt.pick_action(s)
        next_s,r,is_terminal = agnt.take_step(env,a)
        agnt.update_q_table(s,a,next_s,r,is_terminal)
        if is_terminal==True:
            break
        if gui is not None:
            gui.update(episode_number,env.agent_state,agnt.q_table,agnt.actions)
            user_input = gui.check_input()
            grid_positions_clicked = gui.check_mouse()
            for index in range(len(grid_positions_clicked)):
                if (grid_positions_clicked[index] not in GRID_OBSTACLES) and (grid_positions_clicked[index] not in GRID_TERMINALS) and (grid_positions_clicked[index] not in GOAL):
                    GRID_OBSTACLES.append(grid_positions_clicked[index])
                elif(grid_positions_clicked[index] in GRID_OBSTACLES):
                    GRID_OBSTACLES.remove(grid_positions_clicked[index])
                env.redefine(obstacles=GRID_OBSTACLES)
                gui.reset_frame(goal=GOAL, obstacles=GRID_OBSTACLES, terminals=GRID_TERMINALS)
            if user_input == ord('s'):
                gui = None
    agnt.save_learned_model(SAVE_FILENAME)

if __name__ == '__main__':
    gui = GUI(dimensions=GUI_DIMENSION, grid=GRID_SIZE, goal=GOAL, terminals=GRID_TERMINALS, obstacles=GRID_OBSTACLES, mouse_detect=True)
    env = Environment(gui=gui,grid=GRID_SIZE,goal=GOAL,terminals=GRID_TERMINALS,obstacles=GRID_OBSTACLES)
    agent = Agent(grid=GRID_SIZE, actions=ACTIONS, epsilon=EPSILON, gamma=GAMMA, alpha=ALPHA, load_learned=LOAD_AND_SAVE_LEARNED_MODEL)
    for episode in range(N_EPISODES):
        if episode%SHOW_STEP == 0:
            print 'Running episode: ',episode,'/',N_EPISODES
        if episode%GUI_UPDATE_STEP == 0:
            print 'Press \'s\' to skip visualization of this episode'
            run_episode(episode,agent,env,gui=gui)
        else:
            run_episode(episode,agent,env)
