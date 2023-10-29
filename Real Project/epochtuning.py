import time
import random

class EpochTuning:

    

    def set_epoch(self,time):
        return int((time * random.randint(1, 200))/100)