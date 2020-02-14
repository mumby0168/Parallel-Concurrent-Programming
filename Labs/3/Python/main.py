import logging
import time
import threading


def threadFunction(id):
    logging.info("thread function: %d started", id)
    time.sleep(2)
    logging.info("thread function %d ended", id)



if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s:%(message)s",level=logging.INFO,datefmt="%H:%M:%S")
    
    thread = threading.Thread(target=threadFunction, args=(1,))
    thread.start()
