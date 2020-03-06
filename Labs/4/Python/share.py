import time
import logging as logger

class SharedData:
    def __init__(self):
        self.value = 0
    
    def update(self, id):
        local_value = self.value
        time.sleep(1)
        self.value = local_value + 1
        # FIX = self.value = self.value + 1
        #  This is as the local value depends which thread gets there first

    def log(self):
        logger.info("SharedData value = %d", self.value)