from guard import GuardedLock
import threading


class AtomicCounter:
    def __init__(self):
        self._counter = 0
        self._lock = threading.Lock()

    def reset(self):
        GuardedLock(self._lock)
        self._counter = 0

    def inc(self):
        GuardedLock(self._lock)
        self._counter += 1
        return self._counter

    def dec(self):
        GuardedLock(self._lock)
        self._counter -= 1
        return self._counter

    @property
    def counter(self):
        GuardedLock(self._lock)
        return self._counter

    @counter.setter
    def counter(self, value):
        self._counter = value


