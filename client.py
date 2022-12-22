"""
Client class for connecting with server (raspberry pi) which connects with thymio robot.

Classes:  
  Client - sending the action to server (raspberry pi).
"""

import socket

client_dict = {
    0: 'up',
    1: 'down',
    2: 'left',
    3: 'right'
}

class Client():
    """
    Client connecting with server (raspberry pi). Only sends the chosen action.
    """    
    def __init__(self):
        """      
        Initiates the client to connect with thymio-server running in raspberry
        """            
        self.HEADER = 64
        self.PORT = 5050
        self.FORMAT = 'utf-8'
        self.DISCONNECT_MESSAGE = "quit"
        self.SERVER = "10.9.29.50" # change accordignly
        self.ADDR = (self.SERVER, self.PORT)
        
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client.connect(self.ADDR)
        self.robot_moves_time = []
        
    def send(self, msg):
        """
        Sends the message (action)
        
        :param msg: one of the strings: 'up', 'down', 'left', 'right'
        """          
        message = msg.encode(self.FORMAT)
        msg_length = len(message)
        send_length = str(msg_length).encode(self.FORMAT)
        send_length += b' ' * (self.HEADER - len(send_length))
        self.client.send(send_length)
        self.client.send(message)
             
    def send_action(self, action):
        """
        Sends the actions to server and waits for reply until the action is completed succesfully
        
        :param action: one of the ints: 0, 1, 2 or 3 representing an action
        """        
        action = client_dict[action]
        self.send(action)
        robot_move_time = self.client.recv(2048).decode(self.FORMAT) # waiting for reply
        self.robot_moves_time.append(float(robot_move_time)) # time of each robot move
        # print (wait_reply)