import logging
import time
import threading


def threadFunction(id):
    logging.info("thread function: %d started", id)
    time.sleep(2)
    logging.info("thread function %d ended", id)



if __name__ == "__main__":
    time.sleep(2)
    logging.basicConfig(format="%(asctime)s:%(message)s",level=logging.INFO,datefmt="%H:%M:%S")

    numberOfThreads = 8
    threads = []

    for i in range(numberOfThreads):
        logging.info("Creating thread %d", i)
        threads.append(threading.Thread(target=threadFunction, args=(i,)))
        threads[i].start()

    for thread in threads:
        thread.join()
