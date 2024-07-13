import random
import math

from config import *
from utils.game_settings import game_settings

class Target:
    def __init__(self, x, y, w, h, dx, dy):
        w = min(w, game_settings.screen_width)
        h = min(h, game_settings.screen_height)

        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.dx = dx
        self.dy = dy

    def move(self):
        self.x += self.dx
        self.y += self.dy

        if self.x + self.w // 2 > game_settings.screen_width:
            self.x = game_settings.screen_width - self.w // 2
            self.dx = -self.dx

        if self.x - self.w // 2 < 0:
            self.x = self.w // 2
            self.dx = -self.dx

        if self.y + self.h // 2 > game_settings.screen_height:
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
        max_x = game_settings.screen_width - self.w
        max_y = game_settings.screen_height - self.h

        min_visible_area = 0.1  # Minimum visible fraction

        # Calculate range for center coordinates
        center_x_min = self.w / 2 - self.w * (1 - min_visible_area)
        center_x_max = game_settings.screen_width - self.w / 2 + self.w * (1 - min_visible_area)
        center_y_min = self.h / 2 - self.h * (1 - min_visible_area)
        center_y_max = game_settings.screen_height - self.h / 2 + self.h * (1 - min_visible_area)
        
        # Generate random center coordinates
        self.x = random.randint(round(center_x_min), round(center_x_max)) - self.w // 2
        self.y = random.randint(round(center_y_min), round(center_y_max)) - self.h // 2

        # Clamp values to ensure visibility on all sides
        self.x = max(0, min(self.x, max_x))
        self.y = max(0, min(self.y, max_y))


    def randomize_velocity(self):
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

    def adjust_mouse_movement(self, target_x, target_y, game_settings):
        offset_x = target_x - game_settings.screen_x_center
        offset_y = target_y - game_settings.screen_y_center

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