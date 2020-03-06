class GuardedLock:
    def __init__(self, lock):
        self.lock = lock
        self.lock.acquire()

    def __del__(self):
        self.lock.release()