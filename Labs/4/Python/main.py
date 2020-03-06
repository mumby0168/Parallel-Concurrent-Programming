from share import *
import threading
import logging as logger

def threadFunc(data, id):
    data.update(id)
    data.log()

    

if __name__ == "__main__":    
    logger.basicConfig(format="%(asctime)s:%(message)s",level=logger.INFO,datefmt="%H:%M:%S")
    share = SharedData()

    logger.info('Starting program')

    numOfThreads = 6
    threads = []

    for i in range(numOfThreads):
        threads.append(threading.Thread(target=threadFunc, args=(share, i,)))
        threads[i].start()

    for thread in threads:
        thread.join()

    logger.info('ending program')


    