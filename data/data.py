import threading
import queue
import os

from config import *

class Data:
    def __init__(self, delete_prev_data=True):
        self.data_path = './data.txt'
        self.data_queue = queue.Queue()
        self.thread = threading.Thread(target=self.write_to_file, name='Data')
        self.running = True
        self.thread.start()

    def write_to_file(self):
        while self.running:
            target = self.data_queue.get()
            if target is None:
                break
            with open(self.data_path, 'a') as f:
                f.write(
                    f'{target[0]} {target[1]} {target[2]} {target[3]} {target[4]} {target[5]} {target[6]} {target[7]} {target[8]} {target[9]} {target[10]} {target[11]}\n')

    def add_target_data(self, target):
        self.data_queue.put(target)

    def stop(self):
        self.running = False
        self.data_queue.put(None)  # call break
        self.thread.join()
        
data = Data()