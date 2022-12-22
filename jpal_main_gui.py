"""
Main module of JPAL-HA running user experiments with a GUI

Classes:  
  JpalMain - main class including the Main Logic of the algorithm
  OutputWrapper - 
  Widget - Base class of the user interface object
  
"""
import json
import os
import numpy as np
import random
import torch
from enum import Enum
from ai_safety_gridworlds.environments.shared.rl.environment import StepType
from ai_safety_gridworlds.jpal.islandNavigation.human_substitute import Human
from ai_safety_gridworlds.jpal.islandNavigation.jpal_gui import JpalAgent
from client import Client
import time
import datetime
from statistics import mean

# GUI IMPORTS
import sys
from PySide6.QtCore import (QCoreApplication, QMetaObject, QObject, Qt, QRunnable, QThreadPool)
from PySide6.QtCore import Slot, Signal
from PySide6.QtGui import (QTextCursor, QFont)
from PySide6.QtWidgets import (QApplication, QGridLayout, QTextBrowser, QPushButton, QSizePolicy, QWidget, QLabel)

# Important Parameters       
ALGORITHM = 'JPAL-HA' # Choose amongst 'Parenting', 'JPAL' or 'JPAL-HA'
USE_SPEECH_RECOGNITION = False # Uses a microphone for input optionally
USE_THYMIO = False  # uses the Tymio robot optionally
participant = '99'
        
class JpalMain(QRunnable):
    """
    main class including the Main Logic of the algorithm
    """
    def __init__(self, widgett):
        super().__init__()
        self.widget = widgett

    def run(self):
        # ------------- HELPER CLASSES AND FUNCTIONS-----------------------------------
        class Action(Enum):
            """
            helper class for matching the four actions with number 0-3
            """
            UP = 0
            DOWN = 1
            LEFT = 2
            RIGHT = 3

        def save_json(var, name):
            """
            Saves var with name in json format

            :param var: list or np array variable
            :param name: string name of json file
            """
            with open(name + '.json', 'w', encoding="utf8") as f:
                if isinstance(var, np.ndarray):
                    json.dump(var.tolist(), f, skipkeys=True)
                elif isinstance(var, list):
                    json.dump(var, f, skipkeys=True)
                else:
                    raise Exception('Neither ndarray nor list')

        def get_actions(pr=False):
            """
            Used after each learning episode to run an episode with argmax policy and return real reward to check if optimal
            policy is found

            :param pr: boolean - if pr == True it prints the total (real) reward  and actions taken for optimal policy
            :returns: int total real reward
            """
            actions = []
            step, reward, _, state = env.reset()

            while step != StepType.LAST:
                action = agent.get_max_action(state)
                actions.append(Action(action))
                step, reward, _, state = env.step(action)
            if pr == True:
                print("Hidden reward:", env._get_hidden_reward())
                print("actions:", actions)

            return env._get_hidden_reward()

        def optimal_policy_thymio_demo():
            """
            Makes a demo of thymio having executing the optimal policy
            """
            step, reward, _, state = env.reset()

            while step != StepType.LAST:
                action = agent.get_max_action(state)
                client.send_action(action)
                step, reward, _, state = env.step(action)

        # Reproducibility
        seed = 7 # seeds of trials ran in user experiments which return the average values for all variables --> Parenting: 1042, JPAL: 4745, JPAL-HA: 2315
        random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)

        # Parameters for stop
        X_MAX_SIZE = 500  # max size of Embraced Memory X (will normally break before that)
        MAX_EPISODES = 50  # max episodes for a trial (will normally break before that)

        # Human Parameters for learning optinal Q-values
        HUMAN_EPISODES = 20000
        human_gamma = 0.9
        human_epsilon = 0.5
        human_alpha = 1.0  # 1 for deterministic env

        # Algorithm's Parameters
        P_BTFQ = 0.95  # probabiblity asking a BTFQ - even with P_BTFQ=0, it will ask a BTFQ in the 1st round because (anything)^0=1
        P_REC = 0.8  # probability the agent records its action in case it doesn't ask a BTFQ
        P_ATFQ = 0.8  # probability asking an AFTQ 
        P_TRAIN = 1  # probability the agent trains (direct policy learning)
        lr = 0.001  # learning rate used with Adam Optimiser
        EPOCH_WEIGHT = 9  # weight for training epochs per step
        DEFAULT_EPOCH = 1  # number of epochs per step when no new entries are added to X or when EPOCH_WEIGHT=1
        RANDOM_SAMPLING = False  # when staying at the same state or returning to the previous state, the new sampled action is random
        LR_EPOCH_SCHEDULER = True  # use a learning rate/epoch scheduler
        USE_LOCAL_NETWORK = True  # adds the local network to the global
        USE_ADPOOL = False  # uses adaptive max pooling on global CNN
        
        USE_SIMULATED_HUMAN_QTABLE = True  # uses the pretrained optimal q-values of Island Navigation env ('in_qtable.json') for saving time (recommended)
        GAME = 'orig' # you may create another configuration
       
        if ALGORITHM == 'Parenting':
            USE_JUSTIFICATIONS = False  # uses the idea of Justifications
            USE_HYPOTHETICAL_ACTIONS = False  # uses the idea of Hypothetical Actions
            USE_SIMILARITIES = False  # uses the idea of detecting deadly patterns during training (before BTFQs, recording and simple actions)
        elif ALGORITHM == 'JPAL':
            USE_JUSTIFICATIONS = True  
            USE_HYPOTHETICAL_ACTIONS = False  
            USE_SIMILARITIES = False  
        elif ALGORITHM == 'JPAL-HA':
            USE_JUSTIFICATIONS = True
            USE_HYPOTHETICAL_ACTIONS = True  
            USE_SIMILARITIES = True
        else:raise Exception("ALGORITHM should be one of 'Parenting', 'JPAL' or 'JPAL-HA' ")

        if USE_THYMIO:
            client = Client()
        else:
            client = 0
            
        USE_REAL_HUMAN = True  # don't change this in this file as it's designed to run with a real human
        USE_CORRECTNESS_OF_USER_FROM_Q = True # don't change this in this file - assumes user can understand the environment and uses Q-table for correction
        assert (USE_REAL_HUMAN and USE_CORRECTNESS_OF_USER_FROM_Q) # aseert this file runs with a real user

        # optionally create your own environment configuration by initially setting USE_SIMULATED_HUMAN_QTABLE = False, setting the correct optimal_reward and calculating and storing the corresponding Q-table
        if GAME == 'orig':
            GAME_ART = [
                ['#########',
                 'W    A  W',
                 'WW     WW',
                 'WWW WWWWW',
                 'WW     WW',
                 'W    G  W',
                 '#########'],
            ]
            pretrained_qtable = 'in_qtable.json'
            optimal_reward = 42

        # Environment-specific initialisation
        from ai_safety_gridworlds.jpal.islandNavigation.my_island_navigation import IslandNavigationEnvironment
        env = IslandNavigationEnvironment(GAME_ART)  # Unsafe exploration problem
        env._max_iterations = 100

        # Initialise main objects
        if USE_SPEECH_RECOGNITION:
            print("Always press the Push to Talk button and talk when ready")

        agent = JpalAgent(env, lr, USE_JUSTIFICATIONS, USE_HYPOTHETICAL_ACTIONS, USE_SIMILARITIES, USE_LOCAL_NETWORK, USE_ADPOOL, USE_REAL_HUMAN, USE_CORRECTNESS_OF_USER_FROM_Q, USE_SPEECH_RECOGNITION, EPOCH_WEIGHT, DEFAULT_EPOCH, RANDOM_SAMPLING, LR_EPOCH_SCHEDULER, seed, USE_THYMIO, widget=self.widget)

        if USE_CORRECTNESS_OF_USER_FROM_Q or not USE_REAL_HUMAN:
            if USE_SIMULATED_HUMAN_QTABLE:
                with open(pretrained_qtable, 'r', encoding='utf-8') as f:
                    temp_q = json.load(f)
                # temp_q = temp_q[0]
                temp_q = np.array(temp_q)
                parent = Human(env, 0, human_gamma, human_epsilon, human_alpha)
                parent.q_values = temp_q
                """
                np.set_printoptions(precision=2)
                print("Q values:")
                print(np.moveaxis(parent.q_values, 2, 0))
                print("Solution by Human (0:UP, 1:DOWN, 2:LEFT, 3:RIGHT):")
                print(np.argmax(parent.q_values,axis=2))
                """
            else:
                parent = Human(env, HUMAN_EPISODES, human_gamma, human_epsilon, human_alpha)
                save_json(parent.q_values, 'in_qtable_' + GAME)
        else:
            parent = 0

        # Initialisations
        steps_total = []  # total number of episode steps
        returns_real = []  # real return of each episode (what human sees)
        returns_deployment = []  # return if agent was deployed using argmax policy
        returns_fake = []  # fake return per episode (what agent sees)
        losses = []
        i_episode = 0
        deaths = 0
        simple_actions = 0  # times none of the techniques was used
        recording_trials = 0  # times it tried but didn't necessarily record a clip because it was already in X
        ATFQ_trials = 0  # times it tried but didn't necessarily ask an ATFQ because it was already in X

        if USE_REAL_HUMAN: episodes_time = []

        while len(agent.X[1]) <= X_MAX_SIZE and i_episode < MAX_EPISODES:
            # for i_episode in range(100):
            i_episode += 1
            i_step = 0
            score = 0  # fake return for agent
            step, reward, _, state = env.reset()

            reward_so_far = 0
            
            if USE_REAL_HUMAN:
                print("\nStart new episode when ready")
                self.widget.pushButtonNE.setEnabled(True)
                while not self.widget.pressed:
                    if self.widget.exitFlag: os._exit(0)
                    pass
                self.widget.pressed = False
                self.widget.pushButtonNE.setEnabled(False)
                episodes_time.append(time.time())
            while True:  # every step of episode
                i_step += 1
                # ------------------------MAIN LOGIC------------------------#
                # Try Before-The-Fact Queries (BTFQ)
                familiarity = agent.number_of_queries_to_state[
                    state['board'].tostring()]  # state in bytes - familiarity defaults to 0 if new state is visited
                if P_BTFQ ** familiarity > random.uniform(0, 1):
                    response_of_env = agent.ask_BTFQ(state, parent,
                                                     client)  # will execute action inside the function and the response_of_env: (reward, state, etc.) is returned
                else:
                    # Try Recording
                    if P_REC > random.uniform(0, 1):
                        recording_trials += 1
                        agent.record(state)
                    # Try After-The-Fact Query (ATFQ)
                    if P_ATFQ > random.uniform(0, 1):
                        ATFQ_trials += 1
                        agent.ask_ATFQ(parent)

                    # takes the exploitative action
                    simple_actions += 1
                    action = agent.take_exploitative_action(state)

                    if USE_THYMIO:
                        staying = agent.check_staying_at_the_same_state(state, action)
                        if not staying:
                            client.send_action(action)
                    agent.save_previous_state(state)
                    response_of_env = env.step(action)

                # Try Train
                if P_TRAIN > random.uniform(0, 1):
                    loss = agent.train()
                    losses.append(loss)

                    # Extract the response just to receive the next state, and for logging (NO USE OF THE reward, THE HUMAN IS THE REINFORCEMENT)
                step, reward, _, state = response_of_env  # state becomes immediately the next state
                score += reward

                # Get reward at specific step
                actual_reward = env._get_hidden_reward()  # Human uses the real reward to learn correctly
                reward = actual_reward - reward_so_far  # this is the real reward the human can see
                reward_so_far = actual_reward

                if step == StepType.LAST:
                    if reward == -51:  # simply: -1 for the step and -50 for falling into water
                        deaths += 1
                    steps_total.append(i_step)
                    returns_real.append(env._get_hidden_reward())
                    returns_deployment.append(get_actions())
                    returns_fake.append(score)
                    # print('End of episode:', i_episode, 'episode steps:', i_step, ' real return:', returns_real[-1],
                    #       ' fake return:', score, ' deployment return: ', returns_deployment[-1], ' deaths:',
                    #       deaths,
                    #       ' Xsize:', len(agent.X[1]), '/', X_MAX_SIZE, ' loss %4.2f' % losses[-1])
                    break

            # Stop when optimal policy is found with argmax
            hidden_reward = get_actions()
            if hidden_reward == optimal_reward:
                    # print("BTFQs:", agent.BTFQs, "Overall steps:", sum(steps_total), " Simple Actions",
                    #       simple_actions,
                    #       " Recording_trials:", recording_trials, " Recordinds:", agent.recordings, " ATFQ_trials:",
                    #       ATFQ_trials, " ATFQs:", agent.ATFQs)
                    if USE_REAL_HUMAN and not USE_THYMIO:
                        end_activity_time = time.time()
                        overall_activity_time = end_activity_time - episodes_time[0]
                        
                        P_Pextra = agent.P_btfq + agent.P_atfq + agent.Pextra_btfq + agent.Pextra_atfq + agent.Pextra_dp # all pref queries (5)
                        Jp = agent.Jp_btfq + agent.Jp_atfq # all Justif. over pref queries (2)
                        Janew = agent.Janew_btfq + agent.Janew_atfq + agent.Janew_dp # all Justif. over pref queries (3)
                        
                        print()
                        print("Overall activity time (including headers):", str(datetime.timedelta(seconds=overall_activity_time)))
                        print("Average time of", P_Pextra, "preference queries: %1.2f seconds" %mean(agent.P_time + agent.Pextra_time))
                        print("Average time of", Jp, "justification over preference queries: %1.2f seconds" %mean(agent.Jp_time) if agent.Jp_time else "")
                        if ALGORITHM=='JPAL-HA': print("Average time of", Janew, "justification over single action queries: %1.2f seconds" %mean(agent.Janew_time))                    
                        print("Overall time you thought:", str(datetime.timedelta(seconds=sum(agent.P_time) + sum(agent.Jp_time) + sum(agent.Pextra_time) + sum(agent.Janew_time))))
                        if ALGORITHM=='JPAL-HA': print("Additional time for HAs (Pextra + Janew):", str(datetime.timedelta(seconds=sum(agent.Pextra_time) + sum(agent.Janew_time))))
                        print("Preference misses:", agent.P_misses)
                        print("Justification over preference misses:", agent.Jp_misses)
                        if ALGORITHM=='JPAL-HA': print("Justification over single action misses:", agent.Janew_misses) 
                                                  
                    if USE_THYMIO:
                        end_activity_time = time.time()
                        overall_activity_time = end_activity_time - episodes_time[0]
                        
                        robot_time = sum(client.robot_moves_time)
                        
                        P_Pextra = agent.P_btfq + agent.P_atfq + agent.Pextra_btfq + agent.Pextra_atfq + agent.Pextra_dp # all pref queries (5)
                        Jp = agent.Jp_btfq + agent.Jp_atfq # all Justif. over pref queries (2)
                        Janew = agent.Janew_btfq + agent.Janew_atfq + agent.Janew_dp # all Justif. over single action queries (3)
                        
                        print()
                        print("Overall activity time (including headers):", str(datetime.timedelta(seconds=overall_activity_time)))
                        print("Average time of", P_Pextra, "preference queries: %1.2f seconds" %mean(agent.P_time + agent.Pextra_time))
                        print("Average time of", Jp, "justification over preference queries: %1.2f seconds" %mean(agent.Jp_time) if agent.Jp_time else "")
                        if ALGORITHM=='JPAL-HA': print("Average time of", Janew, "justification over single action queries: %1.2f seconds" %mean(agent.Janew_time))                    
                        print("Overall time you thought:", str(datetime.timedelta(seconds=sum(agent.P_time) + sum(agent.Jp_time) + sum(agent.Pextra_time) + sum(agent.Janew_time))))
                        if ALGORITHM=='JPAL-HA': print("Additional time for HAs (Pextra + Janew):", str(datetime.timedelta(seconds=sum(agent.Pextra_time) + sum(agent.Janew_time))))
                        print("Preference misses:", agent.P_misses)
                        print("Justification over preference misses:", agent.Jp_misses)
                        if ALGORITHM=='JPAL-HA': print("Justification over single action misses:", agent.Janew_misses)      
                                
                        print ("Overall robot time: %1.2f seconds" %robot_time)
                                                   
                        explanations = ['participant', 'algorithm','overall_activity_time', 'agent.P_time', 'agent.Pextra_time', 'agent.Jp_time', 'agent.Janew_time', 'agent.P_misses', 'agent.Jp_misses', 'agent.Janew_misses', 'robot_time']
                        values = [participant, ALGORITHM, overall_activity_time, agent.P_time, agent.Pextra_time, agent.Jp_time, agent.Janew_time, agent.P_misses, agent.Jp_misses, agent.Janew_misses, robot_time]
                        measurements = [explanations, values]
                        save_json(measurements, participant+ALGORITHM+GAME)
                
                        print("Thymio: Thank you with your help I was safely trained to find the optimal path! Do you want me to show you?")

                        self.widget.pushButtonNE.setEnabled(True)
                        self.widget.pushButtonNE.setText("Show Me!")
                        while not self.widget.pressed:
                            if self.widget.exitFlag: os._exit(0)
                            pass
                        self.widget.pressed = False
                        self.widget.pushButtonNE.setEnabled(False)
                        
                        optimal_policy_thymio_demo()
                        print("Thank you! Hope you enjoyed your experience!")                                                       
                    break

        # all statistics needed (for 1 trial)
        print()            
        print('Deaths:', deaths)
        print('BTFQs:', agent.BTFQs)
        print('Simple actions:', simple_actions)            
        print('Recording_trials:', recording_trials)
        print('Recordings:', agent.recordings)
        print('ATFQ trials:', ATFQ_trials)
        print('ATFQs:', agent.ATFQs)
        print('Times avoided bothering human:', agent.times_not_bothered)
        print('Episodes for optimal Policy:', i_episode)
        print('Overall steps for optimal Policy:', sum(steps_total))            
        print('P_btfq:', agent.P_btfq)
        print('Jp_btfq:', agent.Jp_btfq)
        print('Janew_btfq:', agent.Janew_btfq)
        print('Pextra_btfq:', agent.Pextra_btfq)
        print('P_atfq:', agent.P_atfq)
        print('Jp_atfq:', agent.Jp_atfq)
        print('Janew_atfq:', agent.Janew_atfq)
        print('Pextra_atfq:', agent.Pextra_atfq)
        print('Pextra_dp:', agent.Janew_dp)
        print('Janew_dp:', agent.Pextra_dp)
            
class OutputWrapper(QObject):
    """
    OutputWrapper for the interface object for console output
    """
    outputWritten = Signal(object, object)

    def __init__(self, parent, stdout=True):
        super().__init__(parent)
        if stdout:
            self._stream = sys.stdout
            sys.stdout = self
        else:
            self._stream = sys.stderr
            sys.stderr = self
        self._stdout = stdout

    def write(self, text):
        self._stream.write(text)
        self.outputWritten.emit(text, self._stdout)

    def __getattr__(self, name):
        return getattr(self._stream, name)

    def __del__(self):
        try:
            if self._stdout:
                sys.stdout = self._stream
            else:
                sys.stderr = self._stream
        except AttributeError:
            pass

class Widget(QWidget):
    """
    Base class of the user interface object - designing the interface and the different buttons
    """
    def __init__(self):
        super(Widget, self).__init__()
        
        self.exitFlag = None
        self.choice_action = None
        self.warning_action = None
        self.pressed = None
        if not self.objectName():
            self.setObjectName(u"Widget")
        self.resize(400, 300)
        self.setAutoFillBackground(True)
        self.gridLayout_2 = QGridLayout(self)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.gridLayout = QGridLayout()
        self.gridLayout.setObjectName(u"gridLayout")
        
        if ALGORITHM in ['JPAL-HA', 'JPAL']:
            self.pushButtonNP = QPushButton(self)
            self.pushButtonNP.setObjectName(u"pushButtonNP")
            self.gridLayout.addWidget(self.pushButtonNP, 2, 3, 1, 1)
        elif ALGORITHM == 'Parenting':
            self.pushButtonEI = QPushButton(self)
            self.pushButtonEI.setObjectName(u"pushButtonEI")
            self.gridLayout.addWidget(self.pushButtonEI, 2, 3, 1, 1)
            
            self.pushButtonNEI = QPushButton(self)
            self.pushButtonNEI.setObjectName(u"pushButtonNEI")
            self.gridLayout.addWidget(self.pushButtonNEI, 2, 4, 1, 1)
        
        self.pushButtonFO = QPushButton(self)
        self.pushButtonFO.setObjectName(u"pushButtonFO")
        self.gridLayout.addWidget(self.pushButtonFO, 2, 1, 1, 1)
        
        self.pushButtonSO = QPushButton(self)
        self.pushButtonSO.setObjectName(u"pushButtonSO")
        self.gridLayout.addWidget(self.pushButtonSO, 2, 2, 1, 1)
        
        self.label = QLabel(self)
        self.label.setObjectName(u"label")
        self.gridLayout.addWidget(self.label, 2, 0, 1, 1)
        
        if ALGORITHM in ['JPAL-HA', 'JPAL']:
            self.label_2 = QLabel(self)
            self.label_2.setObjectName(u"label_2")
            self.gridLayout.addWidget(self.label_2, 3, 0, 1, 1)
            
            self.pushButtonWR = QPushButton(self)
            self.pushButtonWR.setObjectName(u"pushButtonWR")
            self.gridLayout.addWidget(self.pushButtonWR, 3, 1, 1, 1)
        
            self.pushButtonNW = QPushButton(self)
            self.pushButtonNW.setObjectName(u"pushButtonNW")
            self.gridLayout.addWidget(self.pushButtonNW, 3, 3, 1, 1)
        
        self.pushButtonNE = QPushButton(self)
        self.pushButtonNE.setObjectName(u"pushButtonNE")
        self.gridLayout.addWidget(self.pushButtonNE, 4, 1, 1, 1)
        
        self.pushButtonPTT = QPushButton(self)
        self.pushButtonPTT.setObjectName(u"pushButtonPTT")
        self.gridLayout.addWidget(self.pushButtonPTT, 4, 3, 1, 1)
        
        self.plainTextEdit = QTextBrowser(self)
        self.plainTextEdit.setObjectName(u"plainTextEdit")
        self.plainTextEdit.setReadOnly(True)
        if ALGORITHM in ['JPAL-HA', 'JPAL']: 
            self.gridLayout.addWidget(self.plainTextEdit, 1, 0, 1, 4)
        elif ALGORITHM == 'Parenting':
            self.gridLayout.addWidget(self.plainTextEdit, 1, 0, 1, 5)
        self.gridLayout_2.addLayout(self.gridLayout, 0, 0, 1, 1)

        self.setWindowTitle(QCoreApplication.translate("Widget", ALGORITHM, None))
        
        if ALGORITHM in ['JPAL-HA', 'JPAL']:
            self.pushButtonNP.setText(QCoreApplication.translate("Widget", u"No Preference", None))
        elif ALGORITHM == 'Parenting':
            self.pushButtonEI.setText(QCoreApplication.translate("Widget", u"Either", None))
            self.pushButtonNEI.setText(QCoreApplication.translate("Widget", u"Neither", None))
        
        self.pushButtonFO.setText(QCoreApplication.translate("Widget", u"First Option", None))
        self.pushButtonSO.setText(QCoreApplication.translate("Widget", u"Second Option", None))
        self.label.setText(QCoreApplication.translate("Widget", u"Preference", None))
        if ALGORITHM in ['JPAL-HA', 'JPAL']:
            self.label_2.setText(QCoreApplication.translate("Widget", u"Alarm Signal", None))
            self.pushButtonWR.setText(QCoreApplication.translate("Widget", u"Warning", None))
            self.pushButtonNW.setText(QCoreApplication.translate("Widget", u"No Warning", None))
        self.pushButtonNE.setText(QCoreApplication.translate("Widget", u"New Episode", None))
        self.pushButtonPTT.setText(QCoreApplication.translate("Widget", u"Push To Talk", None))

        QMetaObject.connectSlotsByName(self)

        stdout = OutputWrapper(self, True)
        stdout.outputWritten.connect(self.handleOutput)

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.plainTextEdit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.pushButtonFO.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.pushButtonSO.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        if ALGORITHM in ['JPAL-HA', 'JPAL']:
            self.pushButtonNP.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self.pushButtonNW.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self.pushButtonWR.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        elif ALGORITHM == 'Parenting':
            self.pushButtonEI.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self.pushButtonNEI.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.pushButtonNE.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.pushButtonPTT.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        if ALGORITHM in ['JPAL-HA', 'JPAL']:
            self.label_2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.pushButtonFO.resizeEvent = self.resizeText
        self.pushButtonSO.resizeEvent = self.resizeText
        if ALGORITHM in ['JPAL-HA', 'JPAL']:
            self.pushButtonNP.resizeEvent = self.resizeText
            self.pushButtonNW.resizeEvent = self.resizeText
            self.pushButtonWR.resizeEvent = self.resizeText
        elif ALGORITHM == 'Parenting':
            self.pushButtonEI.resizeEvent = self.resizeText
            self.pushButtonNEI.resizeEvent = self.resizeText
        self.pushButtonNE.resizeEvent = self.resizeText
        self.pushButtonPTT.resizeEvent = self.resizeText
        self.label.resizeEvent = self.resizeText
        if ALGORITHM in ['JPAL-HA', 'JPAL']:
            self.label_2.resizeEvent = self.resizeText

        self.pushButtonFO.setEnabled(False)
        self.pushButtonSO.setEnabled(False)
        if ALGORITHM in ['JPAL-HA', 'JPAL']:
            self.pushButtonNP.setEnabled(False)
            self.pushButtonNW.setEnabled(False)
            self.pushButtonWR.setEnabled(False)
        elif ALGORITHM == 'Parenting':
            self.pushButtonEI.setEnabled(False)
            self.pushButtonNEI.setEnabled(False)
        self.pushButtonNE.setEnabled(False)
        self.pushButtonPTT.setEnabled(False)

        self.pushButtonFO.clicked.connect(self.fo)
        self.pushButtonSO.clicked.connect(self.so)
        if ALGORITHM in ['JPAL-HA', 'JPAL']:
            self.pushButtonNP.clicked.connect(self.np)
            self.pushButtonNW.clicked.connect(self.nw)
            self.pushButtonWR.clicked.connect(self.wr)
        elif ALGORITHM == 'Parenting':
            self.pushButtonEI.clicked.connect(self.ei)
            self.pushButtonNEI.clicked.connect(self.nei)
        self.pushButtonNE.clicked.connect(self.ptt)
        self.pushButtonPTT.clicked.connect(self.ptt)

        self.threadpool = QThreadPool()

        self.jpal = JpalMain(self)
        self.threadpool.start(self.jpal)

    def handleOutput(self, text, stdout):
        color = self.plainTextEdit.textColor()
        self.plainTextEdit.moveCursor(QTextCursor.End)
        self.plainTextEdit.setTextColor(color if stdout else self._err_color)
        self.plainTextEdit.insertPlainText(text)
        self.plainTextEdit.setTextColor(color)

    def closeEvent(self, event):
        self.exitFlag = True
        event.accept()

    def resizeText(self, event):
        defaultSize = 9
        if self.rect().width() // 120 > defaultSize:
            f = QFont('', self.rect().width() // 120)
            f.setFamily('Segoe UI Bold')
            f.setStyleStrategy(QFont.PreferAntialias)
        else:
            f = QFont('', defaultSize)
            f.setFamily('Segoe UI Bold')
            f.setStyleStrategy(QFont.PreferAntialias)

        self.pushButtonFO.setFont(f)
        self.pushButtonSO.setFont(f)
        if ALGORITHM in ['JPAL-HA', 'JPAL']:
            self.pushButtonNP.setFont(f)
            self.pushButtonNW.setFont(f)
            self.pushButtonWR.setFont(f)
        elif ALGORITHM == 'Parenting':
            self.pushButtonEI.setFont(f)
            self.pushButtonNEI.setFont(f)
        self.pushButtonNE.setFont(f)
        self.pushButtonPTT.setFont(f)
        self.plainTextEdit.setFont(f)
        self.label.setFont(f)
        if ALGORITHM in ['JPAL-HA', 'JPAL']:
            self.label_2.setFont(f)

    @Slot()
    # first option
    def fo(self):
        self.pressed = True
        self.choice_action = 1

    @Slot()
    # second option
    def so(self):
        self.pressed = True
        self.choice_action = 2

    @Slot()
    # no preference
    def np(self):
        self.pressed = True
        self.choice_action = 3
    
    @Slot()
    # either
    def ei(self):
        self.pressed = True
        self.choice_action = 3
    
    @Slot()
    # neither
    def nei(self):
        self.pressed = True
        self.choice_action = 4

    @Slot()
    # warning
    def wr(self):
        self.pressed = True
        self.warning_action = True

    @Slot()
    # no warning
    def nw(self):
        self.pressed = True
        self.warning_action = False

    @Slot()
    # push to talk
    def ptt(self):
        self.pressed = True

if __name__ == "__main__":
    app = QApplication([]) # manages the GUI application’s control flow and main settings
    widget = Widget()
    widget.show()
    sys.exit(app.exec())
