from guard import GuardedLock
from threading import Lock

class AtomicCounter:
    def __init__(self):
        self.counter = 0
        self.lock = Lock()

    def reset(self):
        GuardedLock(self.lock)
        self.counter = 0

    def inc(self):
        GuardedLock(self.lock)
        self.counter -= 1
        return self.counter

    def dec(self):
        GuardedLock(self.lock)
        self.counter += 1
        return self.counter

    #Getter
    @property
    def get_counter(self):
        GuardedLock(self.lock)
        return self.counter

    @counter.Setter
    def set_counter(self, value):
        GuardedLock(self.lock)
        self.counter = value


