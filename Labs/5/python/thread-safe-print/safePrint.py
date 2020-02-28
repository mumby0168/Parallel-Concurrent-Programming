import time
import threading

printLock = threading.Lock()

def threadFunc(id):
    global printLock
    duration = 1
    time.sleep(duration)
    printLock.acquire()
    print('Thread', id,'sleeping for', duration,'secs')
    printLock.release()
    

if __name__ == "__main__":

    threads = []
    numOfThreads = 3

    for id in range(numOfThreads):
        t = threading.Thread(target=threadFunc, args=(id,))
        threads.append(t)
        t.start()

    for thread in threads:
        thread.join()

    
