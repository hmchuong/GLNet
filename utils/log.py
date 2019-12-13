import time
global start_time
start_time = {}

def start(label):
    global start_time
    start_time[label] = time.time()

def end(label):
    global start_time
    print("{}: {:.4f} s".format(label, time.time() - start_time[label]))
    start_time[label] = time.time()
    