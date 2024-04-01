import numpy as np
class Country(object):
    def __init__(self, name, state):
        self.name=name
        self.current_state = state   # state of conflict intensity/ state in HMM
        self.neigh=[]   # list of neighbors countries
        self.A=np.zeros((3,3))  # transition matrix
        self.nb_conf=0  # number of conflicts involved in

    def toString(self):
        return "Name: " + self.name + "\ Current state: " + str(self.current_state)+"\ Nb of conflicts: "+str(self.nb_conf)+"Transition matrix: "+str(self.A)