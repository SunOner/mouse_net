# data.py
import threading
import queue
import os
from collections import namedtuple

# Assuming 'config.py' has relevant configuration settings
from config import *

# Define a named tuple for better readability
TargetData = namedtuple('TargetData', ['attr1', 'attr2', 'attr3', 'attr4', 'attr5', 
                                      'attr6', 'attr7', 'attr8', 'attr9', 'attr10',
                                      'attr11', 'attr12'])

class Data:
    def __init__(self, delete_prev_data=True, data_path='./data.txt'):
        self.data_path = data_path  # Allow customization of the file path
        self.data_queue = queue.Queue()
        self.thread = threading.Thread(target=self.write_to_file, name='DataWriter')
        self.running = True
        self.thread.start()

        if delete_prev_data:
            try:
                os.remove(self.data_path)
            except FileNotFoundError:
                pass  # File doesn't exist, no need to delete
            except PermissionError as e:
                print(f"Error deleting file: {e}")

    def write_to_file(self):
        while self.running:
            target = self.data_queue.get()
            if target is None:
                break

            try:
                with open(self.data_path, 'a') as f:
                    f.write(" ".join(map(str, target)) + "\n")  # Write as space-separated values
            except (IOError, OSError) as e:
                print(f"Error writing to file: {e}")

    def add_target_data(self, target):
        try:
            data = TargetData(*target)  # Ensure target is a 12-item iterable
            self.data_queue.put(data)
        except TypeError as e:
            print(f"Invalid target data format: {e}")

    def stop(self):
        self.running = False
        self.data_queue.put(None)  # Signal the thread to stop
        self.thread.join()        # Wait for the thread to finish

data = Data()