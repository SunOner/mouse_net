import random
import math

from config import *
from utils.game_settings import game_settings

class Target:
    def __init__(self, x, y, w, h, dx, dy):
        self.w = min(w, game_settings.screen_width)
        self.h = min(h, game_settings.screen_height)

        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.dx = dx
        self.dy = dy

    def move(self):
        self.x += self.dx
        self.y += self.dy

        if self.x + self.w > game_settings.screen_width:
            self.x = game_settings.screen_width - self.w // 2
            self.dx = -self.dx

        if self.x - self.w // 2 < 0:
            self.x = self.w // 2
            self.dx = -self.dx

        if self.y + self.h > game_settings.screen_height:
            self.y = game_settings.screen_height - self.h // 2
            self.dy = -self.dy

        if self.y - self.h // 2 < 0:
            self.y = self.h // 2
            self.dy = -self.dy

    def randomize_size(self):
        max_width = game_settings.screen_width
        max_height = game_settings.screen_height
        self.w = random.randint(4, max_width)
        self.h = random.randint(4, max_height)

    def randomize_position(self):
        # clamp to prevent out of bound targets
        max_x = max(0, game_settings.screen_width - self.w)
        max_y = max(0, game_settings.screen_height - self.h)
        self.x = random.randint(0, max_x)
        self.y = random.randint(0, max_y)

    def randomize_velocity(self):
        from config import Option_gen_speed_x, Option_gen_speed_y
        
        self.dx += random.uniform(Option_gen_speed_x[0], Option_gen_speed_x[1])
        self.dy += random.uniform(Option_gen_speed_y[0], Option_gen_speed_y[1])

        self.dx = max(-1, min(self.dx, 1))
        self.dy = max(-1, min(self.dy, 1))

    def get_velocity(self):
        return self.dx, self.dy

    def get_speed(self):
        return math.sqrt(self.dx**2 + self.dy**2)

    def get_direction(self):
        return math.atan2(self.dy, self.dx)

    def get_direction_degrees(self):
        return math.degrees(self.get_direction())

    def get_position(self):
        return self.x, self.y
    
    def get_size(self):
        return self.w, self.h

    def get_target(self):
        return self.x, self.y, self.w, self.h

    def get_target_center(self):
        return self.x + self.w // 2, self.y + self.h // 2

    def adjust_mouse_movement(self, target_x, target_y, game_settings_module):
        # Calculate offset from center of the target box
        offset_x = target_x - (self.x + self.w // 2) 
        offset_y = target_y - (self.y + self.h // 2)

        degrees_per_pixel_x = game_settings.fov_x / game_settings.screen_width
        degrees_per_pixel_y = game_settings.fov_y / game_settings.screen_height

        mouse_move_x = offset_x * degrees_per_pixel_x

        mouse_dpi_move_x = (mouse_move_x / 360) * \
            (game_settings.mouse_dpi * (1 / game_settings.mouse_sensitivity))

        mouse_move_y = offset_y * degrees_per_pixel_y
        mouse_dpi_move_y = (mouse_move_y / 360) * \
            (game_settings.mouse_dpi * (1 / game_settings.mouse_sensitivity))

        return mouse_dpi_move_x, mouse_dpi_move_y

    def get_center(self):
        return self.x + self.w // 2, self.y + self.h // 2
