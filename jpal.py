"""
Includes

- all mechanisms of Agent:
    - ask BTFQs
    - recording
    - ask AFTQs
    - train

    with optionally using:
        - Justifications
        - Hypothetical Actions
    
- the Model

Classes:  
  GlobalNetwork - the Global Network of the agent
  AgentNetwork - the full Agent Model
  JpalAgent - the class with the main mechanisms: asking BTFQs, Recording, asking ATFQs and Training
                   plus the novel ideas of JPAL-HA: Justifications and Hypothetical Actions
  
Functions:
  get_audio - translates the audio to text (www.techwithtim.net)
"""

NUM_ACTIONS = 4
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import torch.nn.functional as F
from collections import defaultdict
from torch.optim import Adam
import speech_recognition as sr
import itertools
import math
import copy
import random

def get_audio():
    """
    This function is able to detect a users voice, translate the audio to text using
    Google speech recognition and return it. 
    It waits until the user is speaking to start translating/recording the audio 
    source: (source:www.techwithtim.net)
    
    :returns: string translated text
    """ 
    r = sr.Recognizer()
    with sr.Microphone() as source:
        audio = r.listen(source)
        said = ""

        try:
            said = r.recognize_google(audio)
            said = help_dict[said]
            print(said)            
        except Exception as e:
            print("Exception: " + str(e))
    
    return said

help_dict = {
    '1': 'first',
    '2': 'second',
    '3': 'equal',
    'w': 'warning',
    'e': 'no warning',
    's': 'second'
}

class GlobalNetwork(nn.Module):
    """
    The global model of the agent - a CNN with full state (ohe) as input
    """
    def __init__(self, env, USE_ADPOOL):
        """      
        :param USE_ADPOOL: boolean chooses whether to use or not adaptive max pooling for the Global Network
                             to take automatically the state matrix down to 2x2 before the final FC layers
        """        
        super(GlobalNetwork, self).__init__()
        self.USE_ADPOOL = USE_ADPOOL

        # Shape of the input H x W x O (shape(H x W) x objects)
        NumberOfObjects = len(env._value_mapping) # =5 (agent, goal, wall, free road, water)

        if self.USE_ADPOOL:
            self.conv1 = nn.Conv2d(NumberOfObjects, 16, (3, 3), padding=1, stride=1)
            self.conv2 = nn.Conv2d(16, 32, (3, 3), padding=1, stride=1)
            self.conv3 = nn.Conv2d(32, 64, (3, 3), padding=1, stride=1)
            self.conv4 = nn.Conv2d(64, 64, (3, 3), padding=1, stride=1)
    
            self.fc1 = nn.Linear(64*4, 64) 
            self.fc2 = nn.Linear(64, 4)
        else:
            shape = env.observation_spec()['board'].shape # (7,9)

            # Filter depths for the CNN
            filters =[NumberOfObjects] + [16, 32, 64, 64] # same global model as (Frye, 2019)
            self.conv_layers = []

            # footnote 6 in (Frye, 2019): The number of layers depends on the dimensions of the gridworld and are chosen to take the state matrix down to 2x2    
            for i in range(len(filters)-1): 
                if shape[0] > 3 and shape[1] > 3:
                  self.conv_layers.append(nn.Conv2d(filters[i], filters[i+1], kernel_size = 3, stride = 1))
                  self.conv_layers.append(nn.ReLU(inplace=True)) # run ReLu in-place
                  shape = shape[0] - 2, shape[1] - 2
    
                elif shape[0] == 2 and shape[1] == 2: 
                  break
                
                else:
                  # Padding to reduce it to 2x2 
                  if shape[0]==3: padx=1
                  elif shape[0] == 2: padx=2
                  else: padx=0
                  if shape[1]==3: pady=1
                  elif shape[1] == 2: pady=2
                  else: pady=0
                  
                  shape = shape[0] - 2 + padx, shape[1] - 2 + pady
    
                  self.conv_layers.append(nn.ZeroPad2d((pady//2,pady - pady//2, padx//2, padx - padx//2)))
                  self.conv_layers.append(nn.Conv2d(filters[i], filters[i+1], kernel_size = 3, stride = 1))
                  self.conv_layers.append(nn.ReLU(inplace=True)) 
    
            self.conv = nn.Sequential(*self.conv_layers) # Convolutional layers
            self.fc1 = nn.Linear(filters[i]*4, 64) 
            self.fc2 = nn.Linear(64, 4)
                    
    def forward(self, x):
      """
      Forward method of the Global Network
      
      :param x: 4-D torch input to the global network [batch size, Number of Objects, board size x, board size y] (e.g. [#,5,7,9])
      """
      if self.USE_ADPOOL:
          out = self.conv1(x) # [#,16,7,9] 
          out = F.relu(out) # [#,16,7,9]
          out = self.conv2(out) # [#,32,7,9]
          out = F.relu(out) # [#,32,7,9]
          out = self.conv3(out) # [#,64,7,9]
          out = F.relu(out) # [#,64,7,9]        
          out = self.conv4(out) # [#,64,7,9]
          out = F.relu(out) # [#,64,7,9]
     
          out = F.adaptive_max_pool2d(out,(2, 2)) # [#,64,2,2]
          out = torch.flatten(out, 1) # [#,256] 64x2x2=256
          
          out = self.fc1(out) # [#,64]
          out = F.relu(out)
          out = self.fc2(out) # [1,4] - 4 actions
      else:
          s = self.conv(x) # [#,64,2,2] should always end up to 2x2
          s = torch.flatten(s, 1) # [#,256] 64x2x2=256
          s = self.fc1(s) # [#,64]
          s = F.relu(s)
          out = self.fc2(s) # [1,4] - 4 actions

      return out

class AgentNetwork(nn.Module):
    """
    The full Agent Model
    
    Can use whether the average of the Global and Local Net or just the Global Net    
    The average turns out to be better in most cases
    """
    def __init__(self, env, USE_ADPOOL):
        """
        :param - env: Island Navigation environment
        :param - USE_ADPOOL: boolean for adaptive max pooling for the Global Network
        """
        super(AgentNetwork, self).__init__()
        self.global_net = GlobalNetwork(env, USE_ADPOOL)
        self.local_net = nn.Sequential(nn.Linear(4, 64), nn.Linear(64, 4))
    
    def forward(self, state):
        """
        Forward method for the full Agent Network
        
        if state is a tuple of both global and local compoents then both networks are used - else only the global one
        
        :param state: current state of the environment
        """        
        if type(state) == tuple:
            st_global, st_local = state
            out_global = self.global_net(st_global)
            out_local = self.local_net(st_local)
            return (out_global + out_local) / 2
        else:
            return self.global_net(state)
              
class JpalAgent():
    """
    Main Agent class running the agent's different mechanisms
    """  
    def __init__(self, env, lr=0.001, USE_JUSTIFICATIONS=True, USE_HYPOTHETICAL_ACTIONS=True, USE_SIMILARITIES=True, USE_LOCAL_NETWORK=True, USE_ADPOOL=True, USE_REAL_HUMAN=False, USE_CORRECTNESS_OF_USER_FROM_Q=True, USE_SPEECH_RECOGNITION=True, EPOCH_WEIGHT=1.5, DEFAULT_EPOCH=1, RANDOM_SAMPLING=False, LR_EPOCH_SCHEDULER=True, seed=7, USE_THYMIO=False):
        """
        :param env: Island Navigation environment
        :param lr: learning rate of Adam Optimiser (0.001 proved to work better than 0.01)
        :param USE_JUSTIFICATIONS: boolean to use the idea of Justifications
        :param USE_HYPOTHETICAL_ACTIONS: boolean to use Hypothetical Actions
        :param USE_SIMILARITIES: boolean to use Similarities
        :param USE_LOCAL_NETWORK: boolean to addiotionally use adds the local network
        :param USE_ADPOOL: boolean for adaptive max pooling for the Global Network
        :param USE_REAL_HUMAN: boolean for using a real human input for preference queries
        :param USE_SPEECH_RECOGNITION: boolean for using speech recognition from Google for human input
        :param USE_CORRECTNESS_OF_USER_FROM_Q: boolean for using autocorrection of human input given a learnt human_q_table beforehand
        :param EPOCH_WEIGHT: float weight for number of training epochs per step (epochs = weight*(size_X_currently - size_X_previous_step))
        :param DEFAULT_EPOCH: int number of epochs per step when no new entries are added to X or when EPOCH_WEIGHT=1
        :param USE_THYMIO: boolean for sending the chosen action to thymio
        """                         
        self.env = env
        self.lr = lr
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.USE_JUSTIFICATIONS = USE_JUSTIFICATIONS
        self.USE_LOCAL_NETWORK = USE_LOCAL_NETWORK
        self.USE_HYPOTHETICAL_ACTIONS = USE_HYPOTHETICAL_ACTIONS
        self.USE_SIMILARITIES = USE_SIMILARITIES
        self.USE_REAL_HUMAN = USE_REAL_HUMAN
        self.USE_CORRECTNESS_OF_USER_FROM_Q = USE_CORRECTNESS_OF_USER_FROM_Q
        self.USE_SPEECH_RECOGNITION = USE_SPEECH_RECOGNITION
        self.USE_THYMIO = USE_THYMIO
        self.EPOCH_WEIGHT = EPOCH_WEIGHT 
        self.DEFAULT_EPOCH = DEFAULT_EPOCH
        self.RANDOM_SAMPLING = RANDOM_SAMPLING
        self.LR_EPOCH_SCHEDULER = LR_EPOCH_SCHEDULER
        self.numberOfObjects = len(self.env._value_mapping) # e.g. 5
        self.agent_char = self.env._value_mapping['A']
        self.goal_char = self.env._value_mapping['G']
        self.water_char = self.env._value_mapping['W']
        self.goal_pos = self.find_goal_pos(env)
        self.water_pos = self.find_water_pos(env)       
        self.X_len = 0
        
        # counter for number of steps X is not updated, to calculate number of epochs
        self.counter_steps = 0
        # object previous state will be stored here before taking an action
        self.previous_state = 0 

        # list of forbidden combination of 2 catastrophic actions, not to be sampled again in cases of Parenting and JPAL
        self.forbid_comb_catastr_act = np.array([], dtype=np.int64) 
        
        # number of queries been made to each state (i.e. familiarity of state). For Island Navigation env. it can get up to !(Num_actions - 1)=!(4-1)=6
        self.number_of_queries_to_state = defaultdict(int)        
        # Emrbaced Memory X in the form: (state, action1, action2, humanJudgement, justification (only for HA), correct action (only for HA))
        self.X = [], [], [], [], [], []
        # The Recorded Clips memory R where recorded clips are stored
        self.R = [], [], [] # Observation from environment, Action (list for explor/exploit clips)
        # Dangerous Patterns Memory
        self.D = []
        
        self.BTFQs = 0
        self.recordings = 0
        self.ATFQs = 0

        self.policy = AgentNetwork(env, USE_ADPOOL).to(self.device)

        self.optimizer = Adam(self.policy.parameters(), lr=self.lr)
        
        self.times_not_bothered = 0
     
    def convert2ohe(self, board):
        """
        Converts state to One Hot Encoding form for each different object
        
        :param - board: float 2-D numpy of the board
        :returns: 3-D numpy OHE state to feed in Global Network
        """   
        dimensions = self.numberOfObjects, *board.shape # numberOfObjects x H x W e.g. (5,7,9)
        st_global = np.zeros(dimensions)
        for i in range(self.numberOfObjects):
            st_global[i] = (board == i)
        return st_global

    def convert2local(self, state):
        """
        Converts state to local form (fed in Local Network)
        
        :param state: current state of the environment
        :returns: list with the type of object of the 4 neighboring cells of the agent state
        """         
        posx, posy = self.find_agent_pos(state)
        st_local = []

        arr = [-1,0, 1,0, 0,-1, 0,1]
        for i in range(0, 8, 2):
            x, y = posx + arr[i] , posy + arr[i+1]
            st_local.append(state['board'][x][y])
        return st_local 
      
    def find_agent_pos(self, state):
        """
        Finds the position of the agent
        
        :param state: current state of the environment
        :returns: int 2 coordinates of the agent
        """ 
        pos = np.where(state['board'] == self.agent_char)
        if pos[0].size == 0:
            return None
        return pos[0].item(), pos[1].item()   # e.g. (2,5)
    
    def find_goal_pos(self, env):
        """
        Finds the position of the goal cell
        
        :param env: environment object
        :returns: int 2 coordinates of the agent
        """
        _, _, _, state = env.reset()
        pos = np.where(state['board'] == self.goal_char) 
        return pos[0].item(), pos[1].item()   # e.g. (5,5)

    def find_water_pos(self, env):
        """
        Finds the position of the goal cell
        
        :param env: environment object
        :returns: int 2 coordinates of the agent
        """
        _, _, _, state = env.reset()
        pos = np.where(state['board'] == self.water_char) 
        return pos

    def sample_2_distinct_actions(self, state):
        """
        Runs forward method and samples 2 distinct actions from policy
        In case of Parenting or JPAL if a forbidden combination of 2 catastrophic actions was sampled earlier, then a random sampling is done
        
        :param state: current state of the environment
        :returns: int numpy of 2 distinct actions 
        """        
        if type(state) == tuple:
            st_global, st_local = state
            st_global = torch.as_tensor(st_global, dtype=torch.float32).to(self.device).unsqueeze(0)
            st_local = torch.as_tensor(st_local, dtype=torch.float32).to(self.device).unsqueeze(0)
            logits = self.policy((st_global, st_local))
        else:            
            state = torch.as_tensor(state, dtype=torch.float32).to(self.device).unsqueeze(0)
            logits = self.policy(state)
        distr = Categorical(logits=logits.squeeze(0)) # create categorical distribution passing logits
        actions = torch.multinomial(distr.probs, 2, replacement=False).cpu().numpy() # don't sample the same action 2 times        
        if np.all(actions == self.forbid_comb_catastr_act) or np.all(actions == np.flip(self.forbid_comb_catastr_act, 0)): # if sampled again 2 catastrophic ones, make a random choice of two distinct actions
            list_of_actions = list(range(NUM_ACTIONS))
            while True: 
                actions = np.array(random.sample(list_of_actions, 2))
                if not(np.all(actions == self.forbid_comb_catastr_act) or np.all(actions == np.flip(self.forbid_comb_catastr_act, 0))):
                    break
        self.forbid_comb_catastr_act = np.array([], dtype=np.int64) # clear this
        return actions
        
    def sample_1_action(self, state):
        """
        Runs forward method and samples 1 action from policy model
        
        :param state: current state of the environment
        :returns: int 1 sampled action
        """          
        st_global = self.convert2ohe(state['board']) 
        st_global = torch.as_tensor(st_global, dtype=torch.float32).to(self.device).unsqueeze(0)
        if self.USE_LOCAL_NETWORK:
            st_local = self.convert2local(state) 
            st_local = torch.as_tensor(st_local, dtype=torch.float32).to(self.device).unsqueeze(0) 
            logits = self.policy((st_global,st_local))
        else: 
            logits = self.policy(st_global)
        distr = Categorical(logits=logits.squeeze(0))
        return distr.sample().item()

    def sample_1_new_action(self, state, already_seen_actions, force):
        """
        Samples 1 new action from the network other than the already seen actions
        
        :param state: current state of the environment
        :param already_seen_actions: int actions the agent has already sampled
        :param force: boolean for which True forces a sampling but changes the distribution, whereas False might return None if agent persits in specific action
        :returns: int action different than the already seen actions better than the 
        """                
        st_global = self.convert2ohe(state['board']) 
        st_global = torch.as_tensor(st_global, dtype=torch.float32).to(self.device).unsqueeze(0)
        if self.USE_LOCAL_NETWORK:
            st_local = self.convert2local(state) 
            st_local = torch.as_tensor(st_local, dtype=torch.float32).to(self.device).unsqueeze(0)         
            logits = self.policy((st_global, st_local))
        else:
            logits = self.policy(st_global)
        if force == False:                 
            distr = Categorical(logits=logits.squeeze(0))
            for i in range(1000):
                new_act = distr.sample().item()        
                if new_act not in already_seen_actions:
                    return new_act
            return None
        elif force == True:
            probabilities = F.softmax(logits, dim=1).squeeze(0)
            probabilities += 1e-17 # in case the actions other than the already_seen_actions have a 0 prob.
            probabilities[already_seen_actions] = 0 # zero probabilities of already seen actions
            distr = Categorical(probs=probabilities.squeeze(0)) # will normalise the distribution summing to 1
            action = distr.sample().item()                       
            return action             
    
    def get_max_action(self, state):
        """
        Gets action with the highest probability from policy (argmax policy)
        
        :param state: current state of the environment
        :returns: int max action
        """         
        st_global = self.convert2ohe(state['board'])
        st_global = torch.as_tensor(st_global, dtype=torch.float32).to(self.device).unsqueeze(0)
        if self.USE_LOCAL_NETWORK:
            st_local = self.convert2local(state) 
            st_local = torch.as_tensor(st_local, dtype=torch.float32).to(self.device).unsqueeze(0) 
            logits = self.policy((st_global,st_local))
        else: 
            logits = self.policy(st_global)        
        distr = Categorical(logits=logits.squeeze(0))
        return distr.probs.argmax().item()  

    def sample_non_greedy_random_action(self, already_seen_actions):
        """
        Sample an new explorative action randomly
        
        :param already_seen_actions: int actions the agent has already sampled
        :returns: int explorative action sampled randomly
        """        
        list_of_actions = list(range(NUM_ACTIONS))
        for act in already_seen_actions:
            if act in list_of_actions:
                list_of_actions.remove(act)
        if len(list_of_actions)==0: # in case there was a deadly pattern in the exploration case
            return None
        random_explorative_action = random.choice(list_of_actions)
        return random_explorative_action        
    
    def get_max_action_not_in_D(self, state, pattern):
        """
        Gets action with the highest probability from policy (argmax policy) not the deadly pattern
        
        :param state: current state of the environment
        :param pattern: list of actions constituting a deadly pattern (checked that it exists in the underlying state)
        :returns: int greedy action not in D
        """         
        st_global = self.convert2ohe(state['board'])
        st_global = torch.as_tensor(st_global, dtype=torch.float32).to(self.device).unsqueeze(0)
        if self.USE_LOCAL_NETWORK:
            st_local = self.convert2local(state) 
            st_local = torch.as_tensor(st_local, dtype=torch.float32).to(self.device).unsqueeze(0) 
            logits = self.policy((st_global,st_local))
        else: 
            logits = self.policy(st_global)
        logits = logits.squeeze(0)
        log_list = logits.tolist()
        ordered_actions_by_decreasing_probability = sorted(range(len(log_list)), key=lambda k: log_list[k], reverse=True)
        for greedy in ordered_actions_by_decreasing_probability:
            if greedy not in pattern:
                return greedy

    def take_exploitative_action(self, state):
        """
        Taking exploitative action during training, taking into account the deadly patterns
        
        :param state: current state of the environment
        :returns: int exploitative action
        """        
        if self.USE_SIMILARITIES: # check deadly pattern before a simple action
            st_local = self.convert2local(state)
            sort_D = self.D.copy() 
            sort_D.sort(key=len, reverse=True)
            for pattern in sort_D: 
                if [st_local[x] for x in pattern] == [self.water_char]*len(pattern):
                    # print("You are in a pattern case taking an exploitative action")      
                    action = self.get_max_action_not_in_D(state, pattern)
                    break
            else:    
                action = self.get_max_action(state)                   
        else:
            action = self.get_max_action(state)
        return action

    def get_explor_action(self, state, random):
        """
        Gets explorative action, not taking into account dangerous Pattern
        
        :param state: current state of the environment
        :param random: boolen if True then a random non-greedy action is sampled, else policy is sampled
        :returns: int explorative action or max_action if none else was sampled
        """         
        max_action = self.get_max_action(state)
        if random == False:              
            already_seen_actions = [max_action]
            action = self.sample_1_new_action(state, already_seen_actions, force=True)
            if action == None:
                print("Didn't sample a new action during exploration.")
        else:
            action = self.sample_non_greedy_random_action([max_action])
        return action
    
    def get_explorative_action_not_in_D(self, state, pattern, random):
        """
        Gets explorative action, when a deadly pattern is present
        
        :param state: current state of the environment
        :param pattern: list of actions constituting a deadly pattern
        :param random: boolen if True then a random non-greedy action is sampled, else policy is sampled        
        :returns: int explorative action or max_action if none else was sampled
        """  
        max_action = self.get_max_action(state)   
        already_seen_actions = pattern.copy() 
        already_seen_actions.append(max_action)
        if len(already_seen_actions) < NUM_ACTIONS:
            if random == False:
                action = self.sample_1_new_action(state, already_seen_actions, force=True)
                if action == None:
                    print("Didn't sample a new action during exploration.")             
            else:
               action = self.sample_non_greedy_random_action(already_seen_actions)  # 3
        else: 
            action = None # this is only for the initial check of exploration
        return action
            
    def get_explorative_action(self, state, random):
        """
        Gets explorative action, possibly taking into account the Dangerous Pattern Memory
        
        :param state: current state of the environment
        :param random: boolen if True then a random non-greedy action is sampled, else policy is sampled        
        :returns: int explorative action or max_action if none else was sampled
        """                 
        if self.USE_SIMILARITIES:
            st_local = self.convert2local(state)
            sort_D = self.D.copy()
            sort_D.sort(key=len, reverse=True)
            for pattern in sort_D: # pattern should not be [2,1, 2]
                if [st_local[x] for x in pattern] == [self.water_char]*len(pattern):
                    # print("You are in a pattern case taking an explorative action")
                    action = self.get_explorative_action_not_in_D(state, pattern, random)
                    break
            else:    
                action = self.get_explor_action(state, random)                   
        else:
            action = self.get_explor_action(state, random) 
        return action
    
    def convert2move (self, action):
        """
        Converts the action from int type (0-3) to string type
        
        :param - action: action in int type
        :returns: action in string type 
        """          
        if action == 0:
            return "Up"
        elif action == 1:
            return "Down"        
        elif action == 2:
            return "Left"
        elif action == 3:
            return "Right"
    
    def convert2board (self, board):
        """
        Converts board from float type type to char type for better interpretation
        
        'E'=Edge, 'W'=Water,'R'=Road, 'A'=Agent, 'G'=Goal
        
        :param - board: float 2-D numpy of the board
        :returns: char 2-D numpy of the board
        """                 
        new_board = board.astype('str')
        new_board[new_board == '0.0'] = 'E'
        new_board[new_board == '1.0'] = 'R'
        new_board[new_board == '2.0'] = 'A'
        new_board[new_board == '3.0'] = 'W'
        new_board[new_board == '4.0'] = 'G'           
        return new_board
    
    def map_to_mu(self, preference, justification):
        """
        Maps input from human (preference and justfication) to mu in the general case of JPAL
        
        :param preference: one of the strings 'first', 'second' or 'equal'
        :param justification: one of the strings 'warning or 'no warning'
        :returns mu: float between 0 and 1. Represents how much first action is preferred to second: p(a1|s)/(p(a1|s)+p(a2|s))
        """         
        if preference == 'first' and justification == 'no warning':
            mu = 0.75
        elif preference == 'second' and justification == 'no warning':
            mu = 0.25
        elif preference == 'equal':
            mu = 0.5
        elif preference == 'first' and justification == 'warning':
            mu = 1
        elif preference == 'second' and justification == 'warning':
            mu = 0
        return mu
    
    def map_to_mu_without_catastrophe(self, preference):
        """
        Maps input from human (preference) to mu in the general case when it's known a catastrophe is not present in the actions
        
        :param preference: one of the strings 'first', 'second' or 'equal'
        :returns mu: float between 0 and 1. Represents how much first action is preferred to second: p(a1|s)/(p(a1|s)+p(a2|s))
        """         
        if preference == 'first':
            mu = 0.75
        elif preference == 'second':
            mu = 0.25
        elif preference == 'equal':
            mu = 0.5
        return mu    

    def map_to_mu_without_warning(self, preference):
        """
        Maps input from human (preference) to mu in the general case when Parenting (no Justifications) is used
        
        :param preference: one of the strings 'first', 'second' or 'equal'
        :returns mu: float between 0 and 1. Represents how much first action is preferred to second: p(a1|s)/(p(a1|s)+p(a2|s))
        """         
        if preference == 'first':
            mu = 1
        elif preference == 'second':
            mu = 0
        elif preference == 'equal':
            mu = 0.5
        return mu   
    
    def pick_action_without_warning(self, actions, preference):
        """
        Picks the action according to preference of human when the justification is 'no warning' or when Parenting (no Justifications) is used
        
        :param actions: list of 2 int actions
        :param preference: one of the strings 'first', 'second' or 'equal'
        :returns action: int action chosen between actions
        """        
        if preference == 'first':
            action = actions[0]
        elif preference == 'equal':
            s = np.random.binomial(1, 0.5)
            action = actions[s]
        elif preference == 'second':
            action = actions[1]
        return action
    
    def pick_action_with_warning(self, actions, preference):
        """
        Picks the action according to preference of human when the justification is 'no warning' and no Hypothetical Actions is used
        
        :param actions: list of 2 int actions
        :param preference: one of the strings 'first', 'second'
        :returns action: int action chosen between actions                    
        """          
        if preference == 'first':
            action = actions[0]
        elif preference == 'second':
            action = actions[1]
        return action
    
    def give_preference_justification_from_Q(self, q1, q2):
        """
        Returns preference and justication of human according to q-values learnt with RTDP in the general case of JPAL
        
        :param q1: float q-value of 1st action
        :param q2: float q-value of 2nd action
        :returns preference: int action chosen between actions
        :returns justification: one of the strings 'warning or 'no warning'
        """          
        if q1 > 0 and q2 > 0: # if both actions acceptable
            justification = 'no warning' # no warning
            if q1==q2: # equally good
                preference = 'equal'
            elif q1 > q2: # if first action is better
                preference = 'first'
            elif q1 < q2: # if second action is better
                preference = 'second'
        elif q1 < 0 or q2 < 0: # if one of the two actions is catastrophe
            justification = 'warning'
            if q1 > 0 and q2 < 0: # if second action is a catastrophe
                preference = 'first'
            elif q1 < 0 and q2 > 0: # if second action is a catastrophe
                preference = 'second'
            elif q1 < 0 and q2 < 0: # if both actions are catastrophes
                preference = 'equal'
        return preference, justification      
    
    def give_preference_without_warning_from_Q(self, q1, q2):
        """
        Returns preference of human according to q-values learnt with RTDP when justification is 'no warning'
        
        :param q1: float q-value of 1st action
        :param q2: float q-value of 2nd action
        :returns preference: int action chosen between actions
        """          
        if q1==q2: # equally good
            preference = 'equal'
        elif q1 > q2: # if first action is better
            preference = 'first'
        elif q1 < q2: # if second action is better
            preference = 'second'
        return preference
    
    def concatenate_safe_catastr_actions(self, safe_actions, catastrophic_actions):
        """
        Concatenates in a list all the safe and catastrophic actions observed until that time - used when generating Hypothetical Actions
        
        :param safe_actions: int 1 or more (list of ints) safe actions
        :param catastrophic_actions: int 1 or more (list of ints) catastrophic actions
        :returns already_seen_actions: list of int actions, concantenation of safe and catastrophic actions (all actions observed at that time)
        """          
        if type(safe_actions) is int:
            already_seen_actions = [safe_actions].copy()
        elif type(safe_actions) is list:
            already_seen_actions = safe_actions.copy()
        if type(catastrophic_actions) is int:
            already_seen_actions.append(catastrophic_actions)         
        elif type(catastrophic_actions) is list:            
            already_seen_actions.extend(catastrophic_actions)
        return already_seen_actions  
        
    def permute_flat_D(self):
        """
        Creates all the permutations of the patterns in Dangerous Patterns Memory and flats everythin in one
        
        :returns perm_D: list of flatted permutations of patterns
        """         
        perm_D = [list(itertools.permutations(pat)) for pat in self.D]
        perm_D = list(itertools.chain(*perm_D))
        perm_D = [list(elem) for elem in perm_D]
        return perm_D
    
    def check_add_pattern_to_D(self, pattern):
        """
        Checks if a new deadly pattern has been detected and if it does, it adds it in Dangerous Pattern Memory

        :param pattern: list of actions constituting a deadly pattern         
        """          
        perm_D = self.permute_flat_D()
        if pattern not in perm_D:
            self.D.append(pattern)
    
    def check_bothering(self, state, actions, entries_before_X_end):
        """
        Checks if human has been already bothered with a specific query stored in Embraced Memory X
        
        :param state: current state of the environment
        :param actions: list of 2 int actions
        :param entries_before_X_end: until how many entries before the end of X to search
        :returns bothered: boolean indicating if human has been already bothered
        :returns index: index that the bothering was found in X
        """
        st_global = self.convert2ohe(state['board'])
        bothered = False
        index = -1
        for i in range(len(self.X[0]) - entries_before_X_end):
            index +=1
            if self.USE_LOCAL_NETWORK:
                if (np.array_equal(self.X[0][i][0], st_global) and self.X[1][i]==actions[0] and self.X[2][i]==actions[1]) \
                        or (np.array_equal(self.X[0][i][0], st_global) and self.X[1][i]==actions[1] and self.X[2][i]==actions[0]):
                    bothered = True
                    break
            else:
                if (np.array_equal(self.X[0][i], st_global) and self.X[1][i]==actions[0] and self.X[2][i]==actions[1]) \
                        or (np.array_equal(self.X[0][i], st_global) and self.X[1][i]==actions[1] and self.X[2][i]==actions[0]):
                    bothered = True
                    break             
        return bothered, index
    
    def calculate_num_epochs(self):
        """
        The number of training epochs per step is calculated dynamically based on increase of
        the Embraced Memory X. The more new entries added, the more epochs to train.
        if EPOCH_WEIGHT is given 0, then it always runs for just 1 epoch. That it the best setting
        for Parenting.
        
        :returns num_epochs: number of training epochs for that step       
        """        
        if len(self.X[0])==self.X_len or self.EPOCH_WEIGHT==0:
            num_epochs = self.DEFAULT_EPOCH
            self.counter_steps +=1
        else:            
            num_epochs = int(math.ceil((self.EPOCH_WEIGHT*(len(self.X[0]) - self.X_len))))
            self.counter_steps = 0
        self.X_len = len(self.X[0])
        return num_epochs       

    def give_justification_one_action_from_Q(self, q):
        """
        Returns the justification for an action according to the the q-value 
        
        :param q: float q-value
        :returns justification: one of the strings 'warning or 'no warning'      
        """         
        if q > 0:
            justification = 'no warning'
        elif q < 0:
            justification = 'warning'
        return justification   

    def print_board(self, state):
        """
        Prints the current board for interaction with the Human              
        """         
        print("'E'=Edge, 'W'=Water,'R'=Road, 'A'=Agent, 'G'=Goal")
        print(self.convert2board(state['board']))  
            
    def give_justification_one_action_from_real_human(self, state, parent, action, x, y):
        """
        Returns the justification for an action according to a real human input
        I the action taken from a specific state leads to a catastrophe then justification is 'warning', else it's 'no warning'
        
        :param state: current state of the environment
        :param parent: parent object needed for the q-values
        :param action: action in int type
        :param x: int x-coordinate of agent
        :param y: int y-coordinate of agent        
        :returns justification: one of the strings 'warning or 'no warning'      
        """             
        if self.USE_CORRECTNESS_OF_USER_FROM_Q:
            sim_justification = self.give_justification_one_action_from_Q(parent.Q(x, y, action))
        if not self.USE_THYMIO: self.print_board(state)
        print ("Please give justification if Action=" + self.convert2move(action) + " would have been taken")                    
        if self.USE_SPEECH_RECOGNITION:        
            print ("Say: 'WARNING' or 'NO WARNING'")
            while True:
                justification = get_audio()
                if justification in ['warning', 'no warning']:
                    if self.USE_CORRECTNESS_OF_USER_FROM_Q:
                        if justification == sim_justification:
                            break
                        else: print("Are you sure you gave the right answer? Please try again!")
                    else:
                        break
        else:                       
            while True:                
                justification = input("Type: 'w' (Warning), 'e' (No warning) \n")
                if justification in ['w', 'e']:                
                    justification = help_dict[justification]
                    if self.USE_CORRECTNESS_OF_USER_FROM_Q:
                        if justification == sim_justification:
                            break
                        else: print("Are you sure you gave the right answer? Please try again!")
                    else:
                        break
        return justification

    def give_preference_without_warning_from_real_human(self, state, actions, parent, x, y):
        """
        Returns preference according to real human input when justification is 'no warning'
        
        :param state: current state of the environment
        :param actions: list of 2 int actions
        :param parent: parent object needed for the q-values       
        :param x: int x-coordinate of agent
        :param y: int y-coordinate of agent        
        :returns preference: int action chosen between actions      
        """        
        if self.USE_CORRECTNESS_OF_USER_FROM_Q:
            q1 = parent.Q(x, y, actions[0])
            q2 = parent.Q(x, y, actions[1])
            sim_preference = self.give_preference_without_warning_from_Q(q1, q2)
        if not self.USE_THYMIO: self.print_board(state)
        print ("Please give preference")          
        if self.USE_SPEECH_RECOGNITION:              
            print("Say: 'FIRST' ("+self.convert2move(actions[0])+"),'SECOND' ("+self.convert2move(actions[1])+") or 'EQUAL'")
            while True:
                preference = get_audio()
                if preference in ['first', 'second', 'equal']:
                    if self.USE_CORRECTNESS_OF_USER_FROM_Q:
                        if preference == sim_preference:
                            break
                        else: print("Are you sure you gave the right answer? Please try again!")
                    else:
                        break
        else:
            while True:                
                preference = input("Type: '1'("+self.convert2move(actions[0])+"), '2' ("+self.convert2move(actions[1])+") or '3' (equal) \n")
                if preference in ['1', '2', '3']:                
                    preference = help_dict[preference]
                    if self.USE_CORRECTNESS_OF_USER_FROM_Q:
                        if preference == sim_preference:
                            break
                        else: print("Are you sure you gave the right answer? Please try again!")
                    else:
                        break
        return preference
                    
    def give_justification_from_real_human(self, state, actions, parent, x, y):
        """
        Returns justification according to real human input - used in case of equal preference
        
        :param state: current state of the environment
        :param actions: list of 2 int actions 
        :param parent: parent object needed for the q-values      
        :param x: int x-coordinate of agent
        :param y: int y-coordinate of agent       
        :returns justification: one of the strings 'warning or 'no warning'      
        """ 
        if self.USE_CORRECTNESS_OF_USER_FROM_Q:
            q1 = parent.Q(x, y, actions[0])
            q2 = parent.Q(x, y, actions[1])
            sim_preference, sim_justification = self.give_preference_justification_from_Q(q1, q2)                
        if not self.USE_THYMIO: self.print_board(state)                    
        if self.USE_SPEECH_RECOGNITION:
            print ("Please give justification")
            print ("Say: 'WARNING' or 'NO WARNING'")
            while True:
                justification = get_audio()
                if justification in ['warning', 'no warning']:
                    if self.USE_CORRECTNESS_OF_USER_FROM_Q:
                        if justification == sim_justification:
                            break
                        else: print("Are you sure you gave the right answer? Please try again!")
                    else:
                        break               
        else:
            print ("Please give justification")                        
            while True:                
                justification = input("Type: 'w' (Warning), 'e' (No warning) \n")
                if justification in ['w', 'e']:                
                    justification = help_dict[justification]
                    if self.USE_CORRECTNESS_OF_USER_FROM_Q:
                        if justification == sim_justification:
                            break
                        else: print("Are you sure you gave the right answer? Please try again!")
                    else:
                        break 
        return justification

    def give_preference_justification_from_real_human(self, state, actions, parent, x, y):
        """
        Returns preference and justification according to real human input in the general case of JPAL
        
        :param state: current state of the environment
        :param actions: list of 2 int actions
        :param parent: parent object needed for the q-values       
        :param x: int x-coordinate of agent
        :param y: int y-coordinate of agent       
        :returns preference: int action chosen between actions
        :returns justification: one of the strings 'warning or 'no warning'       
        """          
        if self.USE_CORRECTNESS_OF_USER_FROM_Q:
            q1 = parent.Q(x, y, actions[0])
            q2 = parent.Q(x, y, actions[1])
            sim_preference, sim_justification = self.give_preference_justification_from_Q(q1, q2)                
        if not self.USE_THYMIO: self.print_board(state)                    
        if self.USE_SPEECH_RECOGNITION:
            print ("Please give preference")                
            print("Say: 'FIRST' ("+self.convert2move(actions[0])+"),'SECOND' ("+self.convert2move(actions[1])+") or 'EQUAL'")
            while True:
                preference = get_audio()
                if preference in ['first', 'second', 'equal']:
                    if self.USE_CORRECTNESS_OF_USER_FROM_Q:
                        if preference == sim_preference:
                            break
                        else: print("Are you sure you gave the right answer? Please try again!")
                    else:
                        break
            print ("Please give justification")
            print ("Say: 'WARNING' or 'NO WARNING'")
            while True:
                justification = get_audio()
                if justification in ['warning', 'no warning']:
                    if self.USE_CORRECTNESS_OF_USER_FROM_Q:
                        if justification == sim_justification:
                            break
                        else: print("Are you sure you gave the right answer? Please try again!")
                    else:
                        break               
        else:
            print ("Please give preference")
            while True:                
                preference = input("Type: '1'("+self.convert2move(actions[0])+"), '2' ("+self.convert2move(actions[1])+") or '3' (equal) \n")
                if preference in ['1', '2', '3']:                
                    preference = help_dict[preference]
                    if self.USE_CORRECTNESS_OF_USER_FROM_Q:
                        if preference == sim_preference:
                            break
                        else: print("Are you sure you gave the right answer? Please try again!")
                    else:
                        break
            print ("Please give justification")                        
            while True:                
                justification = input("Type: 'w' (Warning), 'e' (No warning) \n")
                if justification in ['w', 'e']:                
                    justification = help_dict[justification]
                    if self.USE_CORRECTNESS_OF_USER_FROM_Q:
                        if justification == sim_justification:
                            break
                        else: print("Are you sure you gave the right answer? Please try again!")
                    else:
                        break 
        return preference, justification

    def check_staying_at_the_same_state(self, state, action):
        """
        Checks if taking action action from state state returns the same state state
        
        :param state: current state of the environment
        :param action: int action        
        :returns: boolean True or False       
        """    
        board = np.copy(state['board'])
        temp_env = copy.deepcopy(self.env)
        next_board = np.copy(temp_env.step(action)[3]['board']) # assuming the agent can sense the environment if taking an action (no need to take it)

        if np.all(board == next_board):
            return True
        else:
            return False
    
    def save_previous_state(self, state):
        """
        Save previous state, because it may be useful at current step
        :param state: current state of the environment
        """
        self.previous_state = copy.deepcopy(state)
    
    def check_returning_to_previous_state(self, state, action):
        """
        Checks if the chosen action brings the agent back to the previous state
        Normally the agent can not sense the next state (as implemented here), but it could have been done with a simple memory rule of L<->R, U<->D
        
        :param state: current state of the environment
        :param action: int action        
        :returns: boolean True or False       
        """          
        previous_board = np.copy(self.previous_state['board'])
        temp_env = copy.deepcopy(self.env)        
        next_board = np.copy(temp_env.step(action)[3]['board'])
        
        if np.all(previous_board == next_board):
            return True
        else:
            return False
            
    def check_thymio_send_action(self, state, action, client):
        """
        Check whether Thymio robot is used and send action if not staying at the same state
        
        :param state: current state of the environment
        :param action: int action
        :param client: Client object used for sending action to thymio              
        """
        if self.USE_THYMIO:
            staying = self.check_staying_at_the_same_state(state, action) 
            if not staying:
                client.send_action(action)
    
    def wrap_up_step(self, state, action, client):
        """
        Complete a step and return the response of the environment
        
        :param state: current state of the environment
        :param action: int action
        :param client: Client object used for sending action to thymio
        :returns response_of_env: response of environment to the action taken            
        """        
        self.check_thymio_send_action(state, action, client)
        self.save_previous_state(state)
        response_of_env = self.env.step(action)
        return response_of_env

    def fill_X0(self, state):
        """
        Fills the first column of the Embraced Memory X with the current state
        
        :param state: current state of the environment
        """                        
        st_global = self.convert2ohe(state['board'])
        if self.USE_LOCAL_NETWORK: 
            st_local = self.convert2local(state)
            self.X[0].append((st_global, st_local))
        else: self.X[0].append(st_global)   
        
    def fill_Mem_with_state_and_2_actions(self, state, action1, action2, mem):
        """
        Fills the first 3 columns of X or R with the state, action1 and action2
        
        :param state: state of the environment
        :param action1: int 1st underlying action
        :param action2: int 2nd underlying action
        :param mem: string Embraced memory 'X' or Recorded Clips memory 'R'       
        """
        if mem == 'X':
            self.fill_X0(state)
            self.X[1].append(action1)
            self.X[2].append(action2)            
        elif mem == 'R':
            self.R[0].append(state) # R[0] is filled in a different form than X[0]                   
            self.R[1].append(action1)
            self.R[2].append(action2) 
        
    def is_clip_in_memory(self, state, action1, action2, mem):
        """
        If a the clip (state, action1, action2) is in memory defined by mem, then return True. Otherwise false.
        Both memories can be checked.
        
        :param state: current state of the environment in (ohe, (local)) form
        :param action1: int 1st action to check
        :param action2: int 2nd action to check
        :param mem: one of the strings 'X', 'R' or 'XR' indicating one or both of the memories
        :return: boolean true or false
        """
        st_global = self.convert2ohe(state['board'])
        if self.USE_LOCAL_NETWORK:
            st_local = self.convert2local(state)        
            state_ = (st_global, st_local)
        else:
            state_ = st_global
            
        if mem == 'X':
            X_list = list(zip(*self.X[0:3]))
            for x in X_list:
                if np.all(state_[0]==x[0][0]):
                    if (action1==x[1] and action2==x[2]) or (action1==x[2] and action2==x[1]):
                        return True
            return False
        elif mem == 'R':
            R_list = list(zip(*self.R))                    
            for r in R_list:            
                if np.all(state_[0]==r[0][0]):
                    if (action1==r[1] and action2==r[2]) or (action1==r[2] and action2==r[1]):
                        return True
            return False
        elif mem == 'XR':
            X_list = list(zip(*self.X[0:3]))
            for x in X_list:
                if np.all(state_[0]==x[0][0]):
                    if (action1==x[1] and action2==x[2]) or (action1==x[2] and action2==x[1]):
                        return True
            R_list = list(zip(*self.R))                    
            for r in R_list:            
                if np.all(state['board']==r[0]['board']):
                    if (action1==r[1] and action2==r[2]) or (action1==r[2] and action2==r[1]):
                        return True
            return False
        else: raise Exception("mem should be 'X', 'R' or 'XR'") 
                    
    def generate_hypothetical_actions(self, preference, actions, state, parent, x, y, counter_actions_X5):
        """
        Investigates the region around a catastrophe more by generating hypothetical actions that
            - fills the Embraced Memory with all combinations of actions detected
            - possibly detects a deadly pattern (actions leading to a catastrophe) and adds it to Dangerous Pattern Memory
            - returns a possibly better (safe) action

        :param preference: preference between the two actions (at least one is a catastrophe)
        :param actions: int the 2 BTFQs, but also can be a deadly pattern of e.g. 3 actions when called from the point of checking for deadly patterns
        :param state: current state of the environment
        :param parent: parent object needed for the q-values
        :param x: int x-coordinate of agent
        :param y: int y-coordinate of agent
        :param counter_actions_X5: counter for how many entries will be added to X[5], i.e. the column representing the correct action in the Embraced Memory X
        :returns: action to be taken
        """
        if preference == 'equal':
            safe_actions = []
            catastrophic_actions = [x for x in actions] # in the case of a dangerous pattern detected from the beginning, it could constitute of more than two actions
            self.check_add_pattern_to_D(catastrophic_actions)
        elif preference == 'first':
            action = actions[0]
            safe_actions = [actions[0]]
            catastrophic_actions = [actions[1]]
        elif preference == 'second':
            action = actions[1]
            safe_actions = [actions[1]]
            catastrophic_actions = [actions[0]]
        while len(safe_actions)<2 and len(safe_actions)+len(catastrophic_actions)<NUM_ACTIONS:
            already_seen_actions = self.concatenate_safe_catastr_actions(safe_actions, catastrophic_actions)
            new_action = self.sample_1_new_action(state, already_seen_actions, force=True)
            if self.USE_REAL_HUMAN:
                print("Generating Hypothetical Action...")
                justification_new_action = self.give_justification_one_action_from_real_human(state, parent, new_action, x, y)
            else:
                justification_new_action = self.give_justification_one_action_from_Q(parent.Q(x, y, new_action))
            if justification_new_action == 'warning':                                   
                for catastrophe in catastrophic_actions:
                    if not self.is_clip_in_memory(state, catastrophe, new_action, mem='X'):
                        self.number_of_queries_to_state[state['board'].tostring()] += 1
                        self.fill_Mem_with_state_and_2_actions(state, catastrophe, new_action, mem='X')
                        self.X[3].append(0.5) # equal
                        self.X[4].append(justification_new_action)
                        counter_actions_X5 +=1
                for safe in safe_actions:
                    if not self.is_clip_in_memory(state, safe, new_action, mem='X'):
                        self.number_of_queries_to_state[state['board'].tostring()] += 1
                        self.fill_Mem_with_state_and_2_actions(state, safe, new_action, mem='X')
                        self.X[3].append(1) # equal
                        self.X[4].append(justification_new_action)
                        counter_actions_X5 +=1                    
                catastrophic_actions.append(new_action)
                self.check_add_pattern_to_D(catastrophic_actions)                            
            elif justification_new_action == 'no warning':
                action = new_action # in case there is no 2nd safe action that can be sampled
                for catastrophe in catastrophic_actions:
                    if not self.is_clip_in_memory(state, catastrophe, new_action, mem='X'):
                        self.number_of_queries_to_state[state['board'].tostring()] += 1
                        self.fill_Mem_with_state_and_2_actions(state, catastrophe, new_action, mem='X')
                        self.X[3].append(0)
                        self.X[4].append(justification_new_action)
                        counter_actions_X5 +=1
                for safe in safe_actions:
                    # Check first if this combination was asked and saved in Embraced Memory before asking human
                    bothered, index = self.check_bothering(state, [safe, new_action], entries_before_X_end=counter_actions_X5)                    
                    if bothered:
                        self.times_not_bothered +=1                                         
                        # we just need to pick an action now (no new entry in X)
                        if self.X[3][index]==0.5 and self.X[4][index]=='no warning':
                            s = np.random.binomial(1, 0.5)
                            action = self.X[s+1][index]
                        else: 
                            action = self.X[5][index] # stored action
                    else:
                        self.number_of_queries_to_state[state['board'].tostring()] += 1
                        self.fill_Mem_with_state_and_2_actions(state, safe, new_action, mem='X')                    
                        if self.USE_REAL_HUMAN:
                            print("Asking for preference of hypothetical actions...")
                            preference = self.give_preference_without_warning_from_real_human(state, [safe, new_action], parent, x, y)
                        else:
                            q1 = parent.Q(x, y, safe)
                            q2 = parent.Q(x, y, new_action)
                            assert(q1>0 and q2>0)
                            preference = self.give_preference_without_warning_from_Q(q1, q2)
                        mu = self.map_to_mu_without_catastrophe(preference)                    
                        self.X[3].append(mu)
                        self.X[4].append(justification_new_action)
                        counter_actions_X5 +=1
                        action = self.pick_action_without_warning([safe, new_action], preference)
                safe_actions.append(new_action) # currently it was not needed like this. Could be used in case a more complex condition between the number of safe and catastrophic actions      
        self.X[5].extend([action]*counter_actions_X5) # for all those combinations, the action to be taken if a query about those is asked, is the best found action from that state
        return action                      
                                 
    def ask_BTFQ(self, state, parent, client):
        """
        Uses guidance from parent by possibly asking a BTFQ and appending example to Embraced Memory X.
        It also returns the response of the environment and the number of the BTFQs 
        
        In case the parent has been bothered with the same question, the agent continues with parent's last answer
        Parenting and JPAL(no Hypothetical Actions) have the weakness when two catastrophic
        actions are stored in X. Then one of them is sampled leading to a catastrophe. JPAL-HA solves
        this problem by sorting the justification and the correct action from that state.
        
        :param state: current state of the environment
        :param parent: parent object needed for the q-values
        :param client: Client object used for sending action to thymio
        :returns: response of environment object
        """      
        x, y = self.find_agent_pos(state)

        # Sample BTFQs                      
        sample_again = True
        st_global = self.convert2ohe(state['board'])        
        while sample_again: # may sample again only in Parenting or JPAL cases
            sample_again = False             
            if self.USE_LOCAL_NETWORK:
                st_local = self.convert2local(state)
                actions = self.sample_2_distinct_actions((st_global, st_local))                    
            else:
                actions = self.sample_2_distinct_actions(st_global)
                
            # Check if agent has already bothered the parent with the same exact query (i.e. if the combination (state, action0, action1) already exists in X)
            checked_actions = []
            bothered, index = self.check_bothering(state, actions, entries_before_X_end=0)
            while bothered:
                self.times_not_bothered +=1
                                
                # pick the action the parent had chosen previously instead of sampling
                if self.USE_HYPOTHETICAL_ACTIONS:
                    if self.X[3][index]==0.5 and self.X[4][index]=='no warning':
                        s = np.random.binomial(1, 0.5)
                        action = self.X[s+1][index]
                    else: 
                        action = self.X[5][index] # stored action
                else:
                    if self.X[3][index] > 0.5: 
                        action = self.X[1][index]
                    elif self.X[3][index] < 0.5: 
                        action = self.X[2][index]
                    else: 
                        s = np.random.binomial(1, 0.5)
                        action = self.X[s+1][index]

                # Check if staying at the same state (hitting wall) or staying returning to same state (assuming deterministic environment).
                # If either, sample a new action, create a a pair with previously preferred action, check bothering in X and continue the same way.
                # If none, take the chosen action. 
                checked_actions.extend([self.X[1][index],self.X[2][index]])
                checked_actions = list(set(checked_actions))
                
                staying = self.check_staying_at_the_same_state(state, action)
                returning = self.check_returning_to_previous_state(state, action)             
                if staying or returning:
                    if len(checked_actions) == NUM_ACTIONS: # there's nothing more to sample so pick the action you think is best
                        response_of_env = self.wrap_up_step(state, action, client)                    
                        return response_of_env  # number of BTFQs remains the same
                    if not self.RANDOM_SAMPLING:
                        new_action = self.sample_1_new_action(state, checked_actions, force=True) 
                    else:
                        new_action = self.sample_non_greedy_random_action(checked_actions) # unstuck from cases of 3 equal actions, when the 4th is the correct, e.g. (r,c)=(3,4) quicker
                    checked_actions.append(new_action)
                    actions = [action, new_action]
                    bothered, index = self.check_bothering(state, actions, entries_before_X_end=0)
                else:
                    response_of_env = self.wrap_up_step(state, action, client)                   
                    return response_of_env  # number of BTFQs remains the same
                
            # Check deadly patterns before BTFQ
            if self.USE_SIMILARITIES and self.USE_HYPOTHETICAL_ACTIONS:
                st_local = self.convert2local(state) # e.g. in (2,2)-->[1, 3, 3, 1]
                sort_D = self.D.copy() # e.g. self.D = [[1,2], [0,2], [1,2,3]] - all it needs is sorting list by decreasing length as only 1 pattern is enough if it's the longest (it will return after that)
                sort_D.sort(key=len, reverse=True)
                for pattern in sort_D: 
                    if [st_local[x] for x in pattern] == [self.water_char]*len(pattern): # attention, the pattern can be more than 2 actions!
                        # print("You are in a new unexplored state that a pattern is involved")
                        # update X with 0.5's between the pattern's actions
                        counter_actions_X5 = 0 # counter for how many times the correct action will be added in X[5] at the end of generate_hypothetical_actions()
                        for comb in itertools.combinations(pattern, 2):
                            if not self.is_clip_in_memory(state, comb[0], comb[1], mem='X'):
                                # print(comb)
                                self.number_of_queries_to_state[state['board'].tostring()] += 1
                                self.fill_Mem_with_state_and_2_actions(state, comb[0], comb[1], mem='X')
                                self.X[3].append(0.5)
                                self.X[4].append('warning')
                                counter_actions_X5 +=1
                        action = self.generate_hypothetical_actions('equal', pattern, state, parent, x, y, counter_actions_X5)
                        response_of_env = self.wrap_up_step(state, action, client)
                        return response_of_env

            self.number_of_queries_to_state[state['board'].tostring()] += 1        
            self.BTFQs +=1
                   
            self.fill_Mem_with_state_and_2_actions(state, actions[0], actions[1], mem='X')                                                               
            if self.USE_JUSTIFICATIONS:
                if self.USE_REAL_HUMAN:
                    preference, justification = self.give_preference_justification_from_real_human(state, actions, parent, x, y)
                else:
                    q1 = parent.Q(x, y, actions[0])
                    q2 = parent.Q(x, y, actions[1])
                    preference, justification = self.give_preference_justification_from_Q(q1, q2)                
                mu = self.map_to_mu(preference, justification)                    
                self.X[3].append(mu)
                
                if self.USE_HYPOTHETICAL_ACTIONS: self.X[4].append(justification)
                                
                if justification == 'no warning':
                    action = self.pick_action_without_warning(actions, preference)
                    if self.USE_HYPOTHETICAL_ACTIONS: self.X[5].append(action)
                elif justification == 'warning':
                    if self.USE_HYPOTHETICAL_ACTIONS:
                        counter_actions_X5 = 1
                        action = self.generate_hypothetical_actions(preference, actions, state, parent, x, y, counter_actions_X5)                                    
                    else: # No Hypothetical Actions
                        if preference == 'equal':
                            self.forbid_comb_catastr_act = np.insert(self.forbid_comb_catastr_act, 0, [self.X[1][-1], self.X[2][-1]])
                            del self.X[0][-1]; del self.X[1][-1]; del self.X[2][-1]; del self.X[3][-1]
                            sample_again = True 
                        else:
                            action = self.pick_action_with_warning(actions, preference)                              
            else: # if using Parenting (no Justifications)
                if self.USE_REAL_HUMAN:
                    preference = self.give_preference_without_warning_from_real_human(state, actions, parent, x, y)
                    if preference == 'equal':
                        justification = self.give_justification_from_real_human(state, actions, parent, x, y)
                        if justification == 'warning':
                            self.forbid_comb_catastr_act = np.insert(self.forbid_comb_catastr_act, 0, [self.X[1][-1], self.X[2][-1]])
                            del self.X[0][-1]; del self.X[1][-1]; del self.X[2][-1]
                            sample_again = True
                        else:
                            mu = self.map_to_mu_without_warning(preference)                    
                            self.X[3].append(mu)
                            action = self.pick_action_without_warning(actions, preference)                             
                    else:
                        mu = self.map_to_mu_without_warning(preference)                    
                        self.X[3].append(mu)
                        action = self.pick_action_without_warning(actions, preference)                 
                else:
                    q1 = parent.Q(x, y, actions[0])
                    q2 = parent.Q(x, y, actions[1])                
                    if  q1==q2 and q1>0: # # if both actions acceptable and equally good (attention on the difference with ATFQ here)
                        self.X[3].append(0.5) # let the network learn that they do are equally good
                        s = np.random.binomial(1, 0.5) # 1 experinment, 0.5 prob for success
                        action = actions[s] # both actions are equally good so 50-50 prob to choose one of them
                    elif q1 > q2: 
                        self.X[3].append(1) # favour action 0
                        action = actions [0] # pick action 0 
                    elif q1 < q2:
                        self.X[3].append(0) # favour action 1
                        action = actions [1] # pick action 1
                    elif q1<0 and q2<0:
                        self.forbid_comb_catastr_act = np.insert(self.forbid_comb_catastr_act, 0, [self.X[1][-1], self.X[2][-1]])
                        del self.X[0][-1]; del self.X[1][-1]; del self.X[2][-1]
                        sample_again = True
        response_of_env = self.wrap_up_step(state, action, client)
        return response_of_env
                    
    def record (self, state):
        """
        Records the exploitative and an explorative action (preferably a random one) from the current state and stores it to Recorded Clips memory.
        It does not take any action yet! It will take the exploitative later. This is literally safe exploration as the explorative action is not taken! 
        
        :param state: current state of the environment       
        """
        familiarity_st = self.number_of_queries_to_state[state['board'].tostring()]
        if familiarity_st == math.factorial(NUM_ACTIONS-1):
            # print("You've learnt everything! There's no reason to record again in this state...")
            return
        
        exploitative_action = self.get_max_action(state)
        
        # better ask a question that might look "funny" but it may be better than the good ones - random=False would not be really exploration and it would have been probably covered on BTFQs                        
        explorative_action = self.get_explorative_action(state, random=True)          
        if explorative_action == None:
            # print("There was no explorative action to sample - all in a Dangerous pattern")
            return 
        
        c = 0
        while self.is_clip_in_memory(state, exploitative_action, explorative_action, mem='XR'): # then this becomes false because [0, None] is not in memory
            explorative_action = self.get_explorative_action(state, random=True) 
            c +=1
            if c==NUM_ACTIONS*50: # works
                # print("You tried to record " + str(NUM_ACTIONS*50) + " times. All the combinations (exploitative, exploratives) are in X or R. No need to record!")
                return
            
        self.recordings +=1
        self.fill_Mem_with_state_and_2_actions(state, exploitative_action, explorative_action, mem='R')

    def ask_ATFQ(self, parent):
        """
        Pops from the beginning of Recorded Clips memory (R) one clip and asks parent an After-the-Fact-Query.
        If query already in Embraced Memory (X) it goes to the next clip of R.
        
        :param parent: parent object needed for the q-values
        """          

        if not all(self.R): # if there are no recorded clips
            return
        state, exploitative_action, explorative_action = self.R[0].pop(0), self.R[1].pop(0), self.R[2].pop(0)
        while self.is_clip_in_memory(state, exploitative_action, explorative_action, mem='X'):
            if not all(self.R): # if R has emptied
                return            
            state, exploitative_action, explorative_action = self.R[0].pop(0), self.R[1].pop(0), self.R[2].pop(0)
                        
        self.number_of_queries_to_state[state['board'].tostring()] += 1 
        self.ATFQs +=1

        if self.USE_THYMIO:
            print("Time for an ATFQ...")
            self.print_board(state)
        
        x, y = self.find_agent_pos(state)                                       

        self.fill_Mem_with_state_and_2_actions(state, exploitative_action, explorative_action, mem='X')             
        if self.USE_JUSTIFICATIONS:
            if self.USE_REAL_HUMAN:
                preference, justification = self.give_preference_justification_from_real_human(state, [exploitative_action, explorative_action], parent, x, y)                
            else:
                q1 = parent.Qstate(state, exploitative_action) 
                q2 = parent.Qstate(state, explorative_action)            
                preference, justification = self.give_preference_justification_from_Q(q1, q2)
            mu = self.map_to_mu(preference, justification)                    
            self.X[3].append(mu) # the case of both being negative here is possible (not good without Hypothetical Actions but possible) - in BTFQs it would sample again
            
            if self.USE_HYPOTHETICAL_ACTIONS: self.X[4].append(justification) 
                         
            if justification == 'no warning':
                action = self.pick_action_without_warning([exploitative_action, explorative_action], preference) # we don't take the action - just checking for X[5]
                if self.USE_HYPOTHETICAL_ACTIONS: self.X[5].append(action)
            elif justification == 'warning':
                if self.USE_HYPOTHETICAL_ACTIONS: # we don't care about the action but we need to check the surrounding area if we see a catastrophe
                    counter_actions_X5 = 1 
                    _ = self.generate_hypothetical_actions(preference, [exploitative_action, explorative_action], state, parent, x, y, counter_actions_X5) 
        else:
            if self.USE_REAL_HUMAN:
                preference = self.give_preference_without_warning_from_real_human(state, [exploitative_action, explorative_action], parent, x, y)                
                mu = self.map_to_mu_without_warning(preference)                    
                self.X[3].append(mu) # the case of both being negative here is possible (not good but possible) - in BTFQs it would sample again               
            else:
                q1 = parent.Q(x, y, exploitative_action)
                q2 = parent.Q(x, y, explorative_action)                                                
                if q1==q2: # the case of both being negative here is possible (not good but possible) - in BTFQs it would sample again
                    self.X[3].append(0.5)
                elif q1>q2: 
                    self.X[3].append(1)
                elif q1<q2: 
                    self.X[3].append(0)             
        
    def train(self):
        """
        Updates the model (improves the policy)        
        Optimisation based on (Bradley-Tery model, 1952) for estimating score functions from paiwise preferences
        
        :returns: float  Binary cross-entropy loss
        """
        # simple rule for defining the number of epochs per step        
        num_epochs = self.calculate_num_epochs()

        # If X is not updated for a bunch of steps, increase lr by 10 (move quick to the other side) and run for many epochs (40)
        if self.LR_EPOCH_SCHEDULER:
            if self.counter_steps == 20:
                num_epochs = 40
                for g in self.optimizer.param_groups:
                    g['lr'] = 10*self.lr
                self.counter_steps = 0
            
        for epoch in range(num_epochs):        
            self.optimizer.zero_grad()
            
            act1 = torch.as_tensor(self.X[1], dtype=torch.long).to(self.device) # action 0
            act2 = torch.as_tensor(self.X[2], dtype=torch.long).to(self.device) # action 1
            targetProb = torch.as_tensor(self.X[3], dtype=torch.float).to(self.device) # human Judgement       
            if self.USE_LOCAL_NETWORK:
                st_global = torch.as_tensor([i[0] for i in self.X[0]], dtype=torch.float32).to(self.device)
                st_local = torch.as_tensor([i[1] for i in self.X[0]], dtype=torch.float32).to(self.device)
                logits = self.policy((st_global, st_local))
            else:
                state = torch.as_tensor(self.X[0], dtype=torch.float32).to(self.device) # starting state 
                logits = self.policy(state)        
            
            # Compute loss
            distr = Categorical(logits=logits)
            # print("probabilities before optimisation: ", distr.probs)
            
            pi1Prob = distr.probs.gather(1,act1.view(-1,1)) # these 2 probs=pis don't sum to 1 as there are 2 more actions
            pi2Prob = distr.probs.gather(1,act2.view(-1,1))
            pi1Prob += 1e-7
            pi2Prob += 1e-7
            
            # either
            bce = nn.BCELoss() # Binary cross entropy loss
            loss = bce((pi1Prob/(pi1Prob+pi2Prob)).squeeze(1), targetProb) #Output: pi1/(pi1+pi2), target: μ in [0-1]
            # or similarly: exactly the same but slower to run (probably some differences because of rounding digits)
            """
            loss=0
            for p1, p2, t in zip(pi1Prob, pi2Prob, targetProb):
                loss += - t * torch.log(p1/(p1+p2)) - (1-t) * torch.log(p2/(p1+p2)) 
                #loss += - t * torch.log(p1/(p1+p2)) - (1-t) * torch.log(1-p1/(p1+p2)) # the same
            loss = loss/len(pi1Prob)
            """
            
            # Backward pass
            loss.backward()
                
            # Update network
            self.optimizer.step()
            
            #print("Epoch %d, loss %4.2f" % (epoch, loss.item()))        
        return loss.item()
