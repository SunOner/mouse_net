import queue
import numpy as np
import threading
import cv2

from config import *
from utils.game_settings import game_settings
from utils.target import Target

class Visualisation(threading.Thread):
    def __init__(self):
        super(Visualisation, self).__init__()
        self.queue = queue.Queue()
        self.cv2_window_name = 'train_mouse_net'
        self.running = True
        self.start()

    def run(self):
        cv2.namedWindow(self.cv2_window_name)
        while self.running:
            image = np.zeros((game_settings.screen_height,
                              game_settings.screen_width, 3), np.uint8)

            try:
                data = self.queue.get(timeout=0.1)  # Get data from the queue
            except queue.Empty:
                continue
            if data is None:
                break

            # Handle different data types (single target or sequence)
            if isinstance(data, Target):  # If it's a single Target object
                target_x, target_y, w, h, dx, dy = data.x, data.y, data.w, data.h, data.dx, data.dy

            elif isinstance(data, tuple):  # If it's a sequence (inputs, targets)
                inputs, targets = data
                target_x, target_y = targets[0], targets[1]  # Extract target_x and target_y from targets
                _, _, _, _, _, _, _, _, w, h = inputs[-1].tolist()  # Extract latest w and h from inputs
            
            if Option_gen_visualise_draw_line:  # If option is enabled
                target = Target(target_x, target_y, w, h, dx, dy)  # Create a Target object for calculations
                x, y = target.adjust_mouse_movement(
                    target_x=target.x + target.w // 2,
                    target_y=target.y + target.h // 2,
                    game_settings_module=game_settings
                ) 
                cv2.line(image, 
                         (int(game_settings.screen_x_center), int(game_settings.screen_y_center)), 
                         (int(target.x + target.w // 2 + x), int(target.y + target.h // 2 + y)), 
                         (0, 255, 255), 2)

            # If it's a single target, data is already a Target object
            cv2.rectangle(image, (int(target_x), int(target_y)), (int(target_x + w), int(target_y + h)), (0, 255, 0), 2)
            cv2.imshow(self.cv2_window_name, image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    def stop(self):
        self.running = False
        self.queue.put(None)
        self.join()
        
visualisation = Visualisation()