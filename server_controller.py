"""
Module run in Raspberry Pi for a real-world experiment with a Thymio robot. Implements the server to receive input from the user and the controller (adapted from https://github.com/lebalz/thympi) to move the robot

Classes:  
  ThymioController - helper class for matching the four actions with number 0-3
  
Functions:
  get_actions - runs an episode with argmax policy after every learning episode to check if optimal policy is found
"""

import dbus
import dbus.mainloop.glib
import os
from time import sleep
import time

import socket

os.system("pkill -n asebamedulla")
aesl_file = 'controller.aesl' 

class ThymioController(object):
    """
    Controls the moves of Thymio. The speed is set at approximately 20cm/s.
    """
    def __init__(self, filename):
        # initialize asebamedulla in background and wait 0.3s to let asebamedulla startup
        os.system("(asebamedulla ser:name=Thymio-II &) && sleep 2")
        
        # init the dbus main loop
        dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)

        # get stub of the aseba network
        bus = dbus.SessionBus()
        asebaNetworkObject = bus.get_object('ch.epfl.mobots.Aseba', '/')

        # prepare interface
        self.asebaNetwork = dbus.Interface(asebaNetworkObject, dbus_interface='ch.epfl.mobots.AsebaNetwork')

        # load the file which is run on the thymio
        self.asebaNetwork.LoadScripts(aesl_file, reply_handler=self.dbusReply, error_handler=self.dbusError)
        
        self.straight_speed = 500
        self.straight_time = 1.3
        self.rotate_90_speed = 250
        self.rotate_90_time = 1.9
        self.straight_time_rot_left = 1.1
        self.straight_time_rot_right = 0.8
                
        self.facing = 'south' # initial orientation of the robot

    def stopAsebamedulla():
        """
        Stop the asebamedulla process. dbus connection will fail after this call
        """
        os.system("pkill -n asebamedulla")

    def run(self):
        """
        Runs event loop
        """
        self.mainLoop()
        
    def dbusReply(self):
        """
        dbus replys can be handled here. Currently ignoring
        """
        pass

    def dbusError(self, e):
        """
        dbus errors can be handled here. Currently only the error is logged. Maybe interrupt the mainloop here
        """
        print('dbus error: %s' % str(e))

    def forward(self, speed, stop_after):
        """
        Executes a forward move of the robot
    
        :param speed: speed - speed of the move
        :param stop_after: number of seconds it stops after the move
        """                   
        tic = time.perf_counter()
        self.asebaNetwork.SendEventName('motor.target', [speed, speed], reply_handler=self.dbusReply, error_handler=self.dbusError)
        sleep(stop_after)
        self.asebaNetwork.SendEventName('motor.target', [0, 0], reply_handler=self.dbusReply, error_handler=self.dbusError)
        self.asebaNetwork.SendEventName('motor.target', [5, 0], reply_handler=self.dbusReply, error_handler=self.dbusError)
        sleep(0.3)
        self.asebaNetwork.SendEventName('motor.target', [0, 0], reply_handler=self.dbusReply, error_handler=self.dbusError)
        toc = time.perf_counter()
        print(f"Lasted {toc - tic:0.4f} seconds")

    def back_up(self, speed, stop_after):
        """
        Executes a backward move
    
        :param speed: speed - speed of the move
        :param stop_after: number of seconds it stops after the move
        """  
        tic = time.perf_counter()
        self.asebaNetwork.SendEventName('motor.target', [-speed, -speed], reply_handler=self.dbusReply, error_handler=self.dbusError)
        sleep(stop_after)
        self.asebaNetwork.SendEventName('motor.target', [0, 0], reply_handler=self.dbusReply, error_handler=self.dbusError)
        toc = time.perf_counter()
        print(f"Lasted {toc - tic:0.4f} seconds")

    def rotate_right(self, speed, stop_after):
        """
        Executes a turn to the right
    
        :param speed: speed - speed of the move
        :param stop_after: number of seconds it stops after the move
        """  
        tic = time.perf_counter()
        self.asebaNetwork.SendEventName('motor.target', [-150, -150], reply_handler=self.dbusReply, error_handler=self.dbusError)
        sleep(0.55)
        self.asebaNetwork.SendEventName('motor.target', [speed, 0], reply_handler=self.dbusReply, error_handler=self.dbusError)
        sleep(stop_after)
        self.asebaNetwork.SendEventName('motor.target', [0, 0], reply_handler=self.dbusReply, error_handler=self.dbusError)
        toc = time.perf_counter()
        print(f"Lasted {toc - tic:0.4f} seconds")
        
    def rotate_left(self, speed, stop_after):
        """
        Executes a turn to the left
    
        :param speed: speed - speed of the move
        :param stop_after: number of seconds it stops after the move
        """  
        tic = time.perf_counter()
        self.asebaNetwork.SendEventName('motor.target', [150, 150], reply_handler=self.dbusReply, error_handler=self.dbusError)
        sleep(0.55)
        self.asebaNetwork.SendEventName('motor.target', [-speed, speed], reply_handler=self.dbusReply, error_handler=self.dbusError)
        sleep(stop_after/2-0.1)
        self.asebaNetwork.SendEventName('motor.target', [0, 0], reply_handler=self.dbusReply, error_handler=self.dbusError)
        toc = time.perf_counter()
        print(f"Lasted {toc - tic:0.4f} seconds")
    
    def execute_move(self, action):
        """
        Executes the decided action
    
        :param action: action to be executed from the robot 
        """          
        #acc = self.asebaNetwork.GetVariable('thymio-II', 'acc') # needed to activate thymio strangely
        
        if self.facing=='south':
            if action=='up':
                self.back_up(speed=self.straight_speed, stop_after=self.straight_time)
                self.facing = 'south'            
            elif action=='down':
                self.forward(speed=self.straight_speed, stop_after=self.straight_time)
                self.facing = 'south'
            elif action=='left':
                self.rotate_right(speed=self.rotate_90_speed, stop_after=self.rotate_90_time)
                self.forward(speed=self.straight_speed, stop_after=self.straight_time_rot_right)
                self.facing = 'west'
            elif action=='right':
                self.rotate_left(speed=self.rotate_90_speed, stop_after=self.rotate_90_time)
                self.forward(speed=self.straight_speed, stop_after=self.straight_time_rot_left)
                self.facing = 'east'
        elif self.facing=='north':
            if action=='up':
                self.forward(speed=self.straight_speed, stop_after=self.straight_time)
                self.facing = 'north'            
            elif action=='down':
                self.back_up(speed=self.straight_speed, stop_after=self.straight_time)
                self.facing = 'north'
            elif action=='left':
                self.rotate_left(speed=self.rotate_90_speed, stop_after=self.rotate_90_time)
                self.forward(speed=self.straight_speed, stop_after=self.straight_time_rot_left)
                self.facing = 'west'
            elif action=='right':
                self.rotate_right(speed=self.rotate_90_speed, stop_after=self.rotate_90_time)
                self.forward(speed=self.straight_speed, stop_after=self.straight_time_rot_right)
                self.facing = 'east'
        elif self.facing=='west':
            if action=='up':
                self.rotate_right(speed=self.rotate_90_speed, stop_after=self.rotate_90_time)
                self.forward(speed=self.straight_speed, stop_after=self.straight_time_rot_right)
                self.facing = 'north'            
            elif action=='down':
                self.rotate_left(speed=self.rotate_90_speed, stop_after=self.rotate_90_time)
                self.forward(speed=self.straight_speed, stop_after=self.straight_time_rot_left)
                self.facing = 'south'
            elif action=='left':
                self.forward(speed=self.straight_speed, stop_after=self.straight_time)
                self.facing = 'west'
            elif action=='right':
                self.back_up(speed=self.straight_speed, stop_after=self.straight_time)
                self.facing = 'west'
        elif self.facing=='east':
            if action=='up':
                self.rotate_left(speed=self.rotate_90_speed, stop_after=self.rotate_90_time)
                self.forward(speed=self.straight_speed, stop_after=self.straight_time_rot_left)
                self.facing = 'north'            
            elif action=='down':
                self.rotate_right(speed=self.rotate_90_speed, stop_after=self.rotate_90_time)
                self.forward(speed=self.straight_speed, stop_after=self.straight_time_rot_right)
                self.facing = 'south'
            elif action=='left':
                self.back_up(speed=self.straight_speed, stop_after=self.straight_time)
                self.facing = 'east'
            elif action=='right':
                self.forward(speed=self.straight_speed, stop_after=self.straight_time)
                self.facing = 'east' 

# Initiate thymio                
thymioController = ThymioController(aesl_file)

# Initiate connection
PORT = 5050
# SERVER = socket.gethostbyname(socket.gethostname())
SERVER = "0.0.0.0"
HEADER = 64
FORMAT = 'utf-8'
DISCONNECT_MESSAGE = "quit"

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
ADDR = (SERVER, PORT)
server.bind(ADDR)
server.listen(1)

print ("Server is waiting to receive data...")
conn, client = server.accept()
print("Connected")

connected = True
while True:
    msg_length = conn.recv(HEADER).decode(FORMAT) # waiting to receive the action here
    if msg_length:
       msg_length = int(msg_length)
       msg = conn.recv(msg_length).decode(FORMAT)    
       if msg == 'quit':
           connected = False
           print (msg)
           break
       print(msg)
       
       tic = time.perf_counter()
       executed = thymioController.execute_move(action=msg)       
       toc = time.perf_counter()
       print(f"Lasted {toc - tic:0.4f} seconds")
       robot_move_time = str(toc-tic)
       
       print(thymioController.facing)
       
       conn.send(robot_move_time.encode(FORMAT))

conn.close()
