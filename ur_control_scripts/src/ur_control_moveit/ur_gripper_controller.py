import sys, os
import socket
import time
import numpy as np

class GripperController:
    """
    Command list
    ACT (action):
        0 Deactive gripper
        1 Activate gripper

    GTO (go to):
        0 Stop
        1 Go to requested position

    ATR (automatic release rutine):
        0 normal
        1 emergency auto-release

    ARD (auto-release direction)
        0 Closing auto-release
        1 Opening auto-release

    FOR (force)
        0 Minimum force
        255 Max force

    SPE (speed)
        0 Minimum speed
        255 Max speed

    OBJ (object detection)
        0 No object detected
        1 Object is detected in opening
        2 Object is detected in closing
        3 Object is loss / dropped

    STA (status)
        0 gripper is in reset
        1 Activation in progress
        2 Not used
        3 Activation is completed

    FLT (Fault status)


    POS (position)
        0 Open
        255 Close
    PRE
        Unknown
    """

    def __init__(self):
        self.HOST = "192.168.1.103" # UR robot IP
        self.PORT = 63352 # port for gripper control

        self.socket_open()

        self.rq_init_gripper()

    # socket
    def socket_open(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.HOST, self.PORT))

    def socket_close(self): 
        self.socket.close()


    # Communication to Gripper
    def setter(self, cmd, value):
        self.socket.send(self.__to_bytes("SET {} {}\n".format(cmd, value)))
        res, _ = self.socket.recvfrom(1024)
        res = res.decode()

        return res=='ack'
        
    def getter(self, cmd):
        self.socket.send(self.__to_bytes("GET {}\n".format(cmd)))
        res , _ = self.socket.recvfrom(1024)
        res = res.decode().strip('\n')

        cmd, value =  res.split()
        return cmd, int(value)


    # Gripper Activation
    def rq_is_gripper_activate(self):
        _, value = self.getter("STA")
        return value==3

    def rq_reset(self):
        # reset gripper and wait for complete
        if self.rq_is_gripper_activate():
            self.setter("ACT", 0)
            self.setter("ATR", 0)
        while self.rq_is_gripper_activate():
            time.sleep(0.01)            

    def rq_activate(self):
        # activate gripper and wait for complete
        if not self.rq_is_gripper_activate():
            # self.setter("ARD", 1)
            self.setter("ACT", 1)
        while not self.rq_is_gripper_activate():
            time.sleep(0.01)            
        time.sleep(0.5)            


    # Gripper motion
    def rq_gripper_open(self):
        self.rq_set_pos(0)
        while not (self.rq_is_object_detected()==1 or self.rq_gripper_position()<10):
            time.sleep(0.01)

    def rq_gripper_close(self):
        self.rq_set_pos(255)
        while not (self.rq_is_object_detected()==2 or self.rq_gripper_position()>245):
            time.sleep(0.01)

    def rq_gripper_move_to(self, position):
        self.rq_set_pos(position)
        while not (self.rq_is_object_detected()==1 or self.rq_is_object_detected()==2 or np.abs(self.rq_gripper_position()-position)<10):
            time.sleep(0.01)


    # Set
    def rq_init_gripper(self, speed=255, force=255):
        print('set speed:', self.setter("SPE", speed))
        print('set force:', self.setter("FOR", force))

    def rq_set_pos(self, position):
        pos =  self.setter("POS", position) 
        goto = self.setter("GTO", 1)
        return pos and goto

    def rq_stop(self):
        return self.setter("GTO", 0)


    # Get
    def rq_gripper_state(self):
        cmd, value = self.getter("STA")
        return value

    def rq_gripper_state_eval(self):
        cmd, value = self.getter("STA")
        if value==0:
            return "gripper is in reset"
        elif value==1:
            return "Activation in progress"
        elif value==2:
            return "Not used"
        elif value==3:
            return "Activation is completed"

    def rq_is_object_detected(self):
        _, value = self.getter("OBJ")
        return value

    def rq_gripper_position(self):
        _, value = self.getter("POS")
        return value


    # local 
    def __to_bytes(self, s):
        return bytes(s.encode())
        

if __name__=="__main__":

    # init
    gc = GripperController()
    print('gripper state:', gc.rq_gripper_state_eval())
    print('gripper activated?:', gc.rq_is_gripper_activate())
    
    # gripper re-activation
    gc.rq_reset()
    gc.rq_activate()

    # set speed and force
    gc.rq_init_gripper(speed=255, force=255)
    print('gripper activated?:', gc.rq_is_gripper_activate())

    time.sleep(1)

    # repeat open and close motion
    gc.rq_gripper_open()
    gc.rq_gripper_close()
    gc.rq_gripper_open()

    gc.socket_close()

