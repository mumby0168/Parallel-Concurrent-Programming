from guard import GuardedLock
import logging as logger
import threading
import time

lock = threading.Lock()


def thread_func(id):
    logger.info('Thread %d starting', id)
    GuardedLock(lock)
    logger.info('Thread %d sleeping', id)
    time.sleep(1)

if __name__ == '__main__':
    logger.basicConfig(format="%(asctime)s:%(message)s", level=logger.INFO, datefmt="%H:%M:%S")
    logger.info('start')

    threads = []

    for i in range(10):
        t = threading.Thread(target=thread_func, args=(id,))
        t.start()
        threads.append(t)



    for t in threads:
        i.join()


