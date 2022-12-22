"""
A module for substituting the real human with a simulated one by computing the optimal Q-values and returning the values and information regarding the human

Classes:
  Human - substitutes the real human with the optimal Q-table and policy 
  Action - helper class for matching the four actions to numbers 0-3
"""

from enum import Enum
from ai_safety_gridworlds.environments.shared.rl.environment import StepType
import numpy as np
import torch
class Human():
    """
    Substitutes the real human with the optimal policy and an (almost) perfect q-table
    """
    def __init__(self, env, episodes = 2000, gamma = 0.9, epsilon = 0.05, alpha = 1.0):
        """
        :param env: Island Navigation environment
        :param episodes: int number of episodes for simulated human to learn
        :param gamma: float discount factor
        :param epsilon: float epsilon of epsilon-greedy policy if not random one used (optional)
        :param alpha: float learning rate of RTDP
        """
        self.env = env
        self.q_values = self.initialiseQvalues(env)
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.episodes = int(episodes)        
        self.agent_char = self.env._value_mapping['A'] # '2.0' in Island Navigation
        self.computeQvalues()
        # self.info_human()

    def initialiseQvalues(self, env):
        """
        Initialises Q-table with 0s
        
        :param env: Island Navigation environment
        :returns: numpy q-table of 0s of the board dimension
        """        
        dim = env.observation_spec()['board'].shape # (7,9)
        dim = *dim, 4 
        return np.zeros(dim)
        
    def Qstate(self, s, a): 
        """ 
        Gives the Q value given state in full form 
        
        :param s: state of the board
        :param a: int action taken
        :returns: float Q-value
        """
        x, y = self._findPos(s)
        return self.Q(x, y, a)

    def Q(self, x, y, action): 
        """ 
        Gives Q given agent state coordinates
        
        :param x: int x-coordinate of agent
        :param y: int y-coordinate of agent
        :param action: int action taken
        :returns: float Q-value        
        """
        return self.q_values[x][y][action]

    def V(self, x, y): #state value function given agent state coordinates
        """ 
        Gives V given agent state coordinates
        
        :param x: int x-coordinate
        :param y: int y-coordinates
        :returns: float state value function       
        """    
        return self.Q(x, y, np.argmax(self.q_values[x][y]))
    
    def Advantage(self, s, a): 
        """
        Calculates advantage function given state in full form
        
        :param s: state in full form
        :param a: int action
        :returns: float advantage function
        """
        x, y= self._findPos(s)
        return self.Q(x,y,a) - self.V(x,y)

    def computeQvalues(self):
        """
        Computes Q-table
        """
        for i in range(self.episodes):            
            states, actions, rewards, termination = self.run1episode() #termination" is 0 if reached GOAL or BAD state and 1 if reached MAX_STEPS
            
            n = len(states)
            q = np.copy(self.q_values)

            for t in range(n-1):
                sx1, sy1 = states[t]
                sx2, sy2 = states[t+1]
                
                # No need for full Q-learning rule because of deterministic environment (i.e. alpha=1)
                q[sx1][sy1][actions[t]] = rewards[t] + self.gamma*self.Q(sx2, sy2, np.argmax(self.q_values[sx2][sy2]))                
                # print(np.moveaxis(q, 2, 0))
            
            # add last reward if terminated because of Goal or Bad state
            if termination.value == 0:
                sx1, sy1 = states[n-1]    
                q[sx1][sy1][actions[n-1]] = rewards[n-1]

            self.q_values = q        
            # print(np.moveaxis(self.q_values, 2, 0))

    def find_agent_pos(self, state):
        """
        Finds the position of the agent
        
        :param state: current state of the environment
        :returns: int 2 coordinates of the agent
        """ 
        pos = np.where(state['board'] == 2.0)
        if pos[0].size == 0:
            return None
        return pos[0].item(), pos[1].item()   # e.g. (2,5)
    
    def run1episode(self):
        """
        Runs one episode using epsilon -greedy policy        
        
        returns: list of tuples of coordinates of states passed at each step
        returns: list of actions taken at each step
        returns: list of rewards received at each step
        returns: extra information and termination reason
        """
        states = []
        actions = []
        rewards = []

        step, reward, _, obs = self.env.reset()
        local_state = self.find_agent_pos(obs)
        states.append(self._findPos(obs))
        # print(obs['board'])
        reward_so_far = 0

        while step != StepType.LAST:
            local_state = self.find_agent_pos(obs)
            #Pick action from e-greedy
            random_for_egreedy = np.random.uniform()        
            if random_for_egreedy > self.epsilon: # exploit
                random_values = self.q_values[local_state] + np.random.uniform(size=4) / 1000      
                action = np.argmax(random_values)  
            else: # explore
                action = action = np.random.choice(4)
            actions.append(action)

            step, reward, _, obs = self.env.step(action) # this will be the fake reward the agent would normmally see
            # print(obs['board'])
            actual_reward = self.env._get_hidden_reward() # Human uses the real reward to learn correctly
            reward = actual_reward - reward_so_far # this is the real reward the human can see
            reward_so_far = actual_reward

            if step != StepType.LAST:
                states.append(self._findPos(obs))
            rewards.append(reward)

        return states, actions, rewards, obs['extra_observations']['termination_reason']

    def _findPos(self, obs): 
        """
        Finds the position coordinates of the agent
        
        :param obs: dict - current observation of environment
        :returns: int - position coordinates of agent current state
        """
        pos = np.where(obs['board'] == self.agent_char) 
        return pos[0].item(), pos[1].item()
    
    def get_actions(self):
        """
        Runs an episode and prints total real reward and actions taken for optimal policy
        """        
        actions = []
        step, reward, _, obs = self.env.reset()
        state = self._findPos(obs)

        while step != StepType.LAST:
            action = np.argmax(self.q_values[state[0]][state[1]])
            actions.append(Action(action))
            step, reward, _, obs = self.env.step(action)
            state = self._findPos(obs)
        
        print("Hidden reward:", self.env._get_hidden_reward())
        print("actions:", actions)
    
    def info_human(self):
        """
        Prints information for the human policy (Q-values and optimal policy)
        """         
        np.set_printoptions(precision=2)
        print("Q values:")
        print(np.moveaxis(self.q_values, 2, 0))
        print("Solution by Human (0:UP, 1:DOWN, 2:LEFT, 3:RIGHT):")
        print(np.argmax(self.q_values,axis=2))
        
class Action(Enum):
    """
    Helper class for matching actions with the numbers 0 to 3
    """
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3 
        
        