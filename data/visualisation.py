# /data/visualisation.py
import queue
import numpy as np
import threading
import cv2
from config import *
from utils.game_settings import game_settings
from utils.target import Target

class Visualisation(threading.Thread):
    def __init__(self, draw_line=True):
        super().__init__()
        self.queue = queue.Queue()
        self.window_name = 'train_mouse_net'
        self.running = True
        self.draw_line = draw_line
        self.start()

    def run(self):
        cv2.namedWindow(self.window_name)
        while self.running:
            image = np.zeros((game_settings.screen_height, game_settings.screen_width, 3), np.uint8)

            try:
                data = self.queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if data is None:
                break

            # To Do Add check for valid 

            try:
                if self.draw_line:
                    x, y = data.adjust_mouse_movement(
                        target_x=data.x + data.w // 2, target_y=data.y + data.h // 2, game_settings=game_settings)
                    cv2.line(image, (int(game_settings.screen_x_center), int(game_settings.screen_y_center)),
                                (int(data.x + data.w // 2 + x), int(data.y + data.h // 2 + y)), (0, 255, 255), 2)

                cv2.rectangle(image, (int(data.x), int(data.y)), (int(data.x + data.w), int(data.y + data.h)), (0, 255, 0), 2)
            except (AttributeError, TypeError) as e:
                print(f"Error drawing visualization: {e}")
                
            cv2.imshow(self.window_name, image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
        cv2.waitKey(1)

    def stop(self):
        self.running = False
        self.queue.put(None)
        self.join()

visualisation = Visualisation()