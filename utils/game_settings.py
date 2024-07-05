import random
from config import *

class GameSettings:
    def __init__(self):
        self.screen_width = Option_screen_width
        self.screen_height = Option_screen_height
        self.screen_x_center = self.screen_width // 2
        self.screen_y_center = self.screen_height // 2
        self.mouse_dpi = Option_mouse_dpi
        self.mouse_sensitivity = Option_mouse_sensitivity
        self.fov_x = Option_fov_x
        self.fov_y = Option_fov_y

    def randomize(self, target):
        prev_screen_width = self.screen_width
        prev_screen_height = self.screen_height
        if Option_random_screen_resolution:
            self.screen_width = random.randint(Option_random_screen_resolution_width[0], Option_random_screen_resolution_width[1])
            self.screen_height = random.randint(Option_random_screen_resolution_height[0], Option_random_screen_resolution_height[1])
        self.screen_x_center = self.screen_width // 2
        self.screen_y_center = self.screen_height // 2
        if Option_random_mouse_dpi:
            self.mouse_dpi = random.randint(Option_random_mouse_dpi_min_max[0], Option_random_mouse_dpi_min_max[1])
        if Option_random_mouse_sensitivity:
            self.mouse_sensitivity = random.uniform(Option_random_mouse_sensitivity_min_max[0], Option_random_mouse_sensitivity_min_max[1])
        if Option_random_fov:
            self.fov_x = random.uniform(Option_random_fov_x[0], Option_random_fov_x[1])
            self.fov_y = random.uniform(Option_random_fov_y[0], Option_random_fov_y[1])
        self.update_target_position(target, prev_screen_width, prev_screen_height)

    def update_target_position(self, target, prev_screen_width, prev_screen_height):
        scale_x = self.screen_width / prev_screen_width
        scale_y = self.screen_height / prev_screen_height
        target.x = int(target.x * scale_x)
        target.y = int(target.y * scale_y)
        target.w = int(target.w * scale_x)
        target.h = int(target.h * scale_y)

game_settings = GameSettings()