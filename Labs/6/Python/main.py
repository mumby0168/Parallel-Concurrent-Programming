from guard import GuardedLock
from atomicCounter import AtomicCounter
import logging
import threading
import time

lock = threading.Lock()


def thread_func(id, counter):
    logging.info('Thread %d starting', id)
    counter.inc()
    logging.info("Thread %d value is: %d", id, counter.counter)
    logging.info('Thread %d sleeping', id)
    time.sleep(1)


if __name__ == '__main__':
    logging.basicConfig(format="%(asctime)s:%(message)s", level=logging.INFO, datefmt="%H:%M:%S")
    logging.info('start')

    threads = []
    a_counter = AtomicCounter()

    for i in range(10):
        t = threading.Thread(target=thread_func, args=(i,a_counter,))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()


