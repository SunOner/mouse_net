import math
import os
import queue
import threading
import cv2
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt


class Data:
    def __init__(self, delete_prev_data=True):
        self.data_path = './data.txt'
        self.data_queue = queue.Queue()
        if delete_prev_data:
            self.delete_data()
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

    def delete_data(self):
        if Option_delete_prev_data:
            try:
                os.remove(self.data_path)
            except:
                pass
            try:
                os.remove('./mouse_net.pth')
            except:
                pass

    def add_target_data(self, target):
        self.data_queue.put(target)

    def stop(self):
        self.running = False
        self.data_queue.put(None)  # call break
        self.thread.join()


class Game_settings:
    def __init__(self):
        self.screen_width = Option_screen_width
        self.screen_height = Option_screen_height
        self.screen_x_center = int(self.screen_width / 2)
        self.screen_y_center = int(self.screen_height / 2)
        self.fov_x = Option_fov_x
        self.fov_y = Option_fov_y
        self.mouse_dpi = Option_mouse_dpi
        self.mouse_sensitivity = Option_mouse_sensitivity

    def randomize(self, target): # add target argument
        if Option_random_screen_resolution:
            prev_screen_width = self.screen_width
            prev_screen_height = self.screen_height
            self.screen_width = random.randint(
                Option_random_screen_resolution_width[0], Option_random_screen_resolution_width[1])
            self.screen_height = random.randint(
                Option_random_screen_resolution_height[0], Option_random_screen_resolution_height[1])
            self.update_target_position(target, prev_screen_width, prev_screen_height) # pass target to update_target_position

        self.screen_x_center = int(self.screen_width / 2)
        self.screen_y_center = int(self.screen_height / 2)

        if Option_random_fov:
            self.fov_x = random.randint(
                Option_random_fov_x[0], Option_random_fov_x[1])
            self.fov_y = random.randint(
                Option_random_fov_y[0], Option_random_fov_y[1])

        if Option_random_mouse_dpi:         
            self.mouse_dpi = random.uniform(Option_random_mouse_dpi_min_max[0], Option_random_mouse_dpi_min_max[1])

        if Option_random_mouse_sensitivity:
            step_size = 0.05
            steps = int((Option_random_mouse_sensitivity_min_max[1] - Option_random_mouse_sensitivity_min_max[0]) / step_size) + 1
            self.mouse_sensitivity = round((random.randint(0,steps) * step_size) + Option_random_mouse_sensitivity_min_max[0], 2)           

    def update_target_position(self, prev_width, prev_height):
        scale_x = self.screen_width / prev_width
        scale_y = self.screen_height / prev_height
        target.x = int(target.x * scale_x)
        target.y = int(target.y * scale_y)
        target.w = min(target.w, self.screen_width)
        target.h = min(target.h, self.screen_height)


class Target:
    def __init__(self, x, y, w, h, dx, dy):
        # Initialize to the center of the rectangle
        self.x = x + w // 2
        self.y = y + h // 2
        self.w = min(w, game_settings.screen_width)
        self.h = min(h, game_settings.screen_height)
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
        max_size = min(game_settings.screen_width, game_settings.screen_height) // 2
        
        # Use NumPy to sample from a distribution (e.g., normal or exponential)
        size = np.random.normal(loc=max_size // 2, scale=max_size // 4)  # Mean size is half, std dev is a quarter of max
        size = int(np.clip(size, 4, max_size))  # Ensure size is within valid range

        self.w = size
        self.h = size

        # Randomly adjust aspect ratio
        aspect_ratio_change = random.uniform(0.5, 2.5)  # Adjust aspect ratio by up to 30%
        if random.random() < 0.5:
            self.w = int(self.w * aspect_ratio_change)
        else:
            self.h = int(self.h * aspect_ratio_change)

        # Ensure dimensions are within screen bounds
        self.w = min(self.w, game_settings.screen_width)
        self.h = min(self.h, game_settings.screen_height)

    def randomize_position(self):
        # Allow for partial out-of-bounds, centered on the target
        max_x = game_settings.screen_width - self.w // 2
        max_y = game_settings.screen_height - self.h // 2

        # Generate a new position within the adjusted range
        self.x = random.randint(self.w // 2, max_x)  
        self.y = random.randint(self.h // 2, max_y)

    def get_center(self):  # Renamed from get_top_left
        return self.x, self.y
    
    def get_top_left(self):  # Added for consistency
        return self.x - self.w // 2, self.y - self.h // 2
    
    def get_bottom_right(self):  # Added for consistency
        return self.x + self.w // 2, self.y + self.h // 2

    def get_top_right(self):  # Added for consistency
        return self.x + self.w // 2, self.y - self.h // 2 

    def get_bottom_left(self):  # Added for consistency
        return self.x - self.w // 2, self.y + self.h // 2
    
    def get_size(self):
        return self.w, self.h

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
                data = self.queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if data is None:
                break

            if Option_gen_visualise_draw_line:
                x, y = target.adjust_mouse_movement(
                    target_x=target.x, target_y=target.y, game_settings=game_settings)
                cv2.line(image, (int(game_settings.screen_x_center), int(
                    game_settings.screen_y_center)), (int(data.x + x), int(data.y + y)), (0, 255, 255), 2)

            # Get center coordinates to draw rectangle
            center_x, center_y = data.x, data.y  

            cv2.rectangle(image, (int(center_x - data.w // 2), int(center_y - data.h // 2)),
                          (int(center_x + data.w // 2), int(center_y + data.h // 2)), (0, 255, 0), 2)
            cv2.imshow(self.cv2_window_name, image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    def stop(self):
        self.running = False
        self.queue.put(None)
        self.join()


class CustomDataset(Dataset):
    def __init__(self, filepath):
        self.data = []
        with open(filepath, 'r') as f:
            for line in f:
                values = list(map(float, line.split()))
                self.data.append((values[:10], values[10:]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        inputs, targets = self.data[idx]
        return torch.tensor(inputs, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)


class Mouse_net(nn.Module):
    def __init__(self):
        super(Mouse_net, self).__init__()
        self.fc1 = nn.Linear(10, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x


def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    sec = seconds % 60

    formatted_time = f"{hours:02}:{minutes:02}:{sec:06.3f}"
    return formatted_time


def train_net():
    save_path = 'models/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print(f'Starting train mouse_net model.\nUsing device: {device}.')
    dataset = CustomDataset(data.data_path)
    dataloader = DataLoader(
        dataset, batch_size=Option_train_batch_size, shuffle=True, pin_memory=True)
    model = Mouse_net().to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=Option_learning_rate)

    epochs = Option_train_epochs
    loss_values = []

    start_time = time.time()
    print(f'Learning rate: {Option_learning_rate}')

    for epoch in range(epochs):
        epoch_losses = []
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
            last_update_time = time.time()

        epoch_loss = np.mean(epoch_losses)
        loss_values.append(epoch_loss)

        train_time = last_update_time - start_time

        print(f'Epoch {epoch + 1}/{epochs}',
              'Loss: {:.5f}'.format(epoch_loss), format_time(train_time))

        if (epoch + 1) % Option_save_every_N_epoch == 0:
            torch.save(model.state_dict(), os.path.join(
                save_path, f'mouse_net_epoch_{epoch + 1}.pth'))
            print(f'Model saved at epoch {epoch + 1}')
        if (epoch == 4) or (epoch == 8) or (epoch == 12) or (epoch == 16) or (epoch == 20) or (epoch == 24) or (epoch == 28): 
            lr = optimizer.param_groups[0]['lr']
            lr = lr / 2
            optimizer.param_groups[0]['lr'] = lr
            print(f'Changing learning rate to {lr }')

    plt.plot(loss_values)
    plt.title('Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

    torch.save(model.state_dict(), 'mouse_net.pth')


def test_net():  # need to be replaced a newer function
    print('Starting testing model...')
    test_dataset = CustomDataset('data.txt')
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = Mouse_net().to(device)
    model.load_state_dict(torch.load('mouse_net.pth', map_location=device))
    model.eval()

    predictions = []
    actuals = []

    with torch.no_grad():
        for inputs, targets in test_dataloader:
            inputs = inputs.to(device)
            prediction = model(inputs)
            predictions.append(prediction.cpu().numpy())
            actuals.append(targets.cpu().numpy())

    mse = np.mean((np.array(predictions) - np.array(actuals))**2)
    print(f"Mean Squared Error on Test Data: {mse}")


def gen_data():
    pbar = tqdm(total=Option_gen_time, desc='Data generation')

    global target
    target = Target(
        x=random.randint(4, game_settings.screen_width - 4),  
        y=random.randint(4, game_settings.screen_height - 4), 
        w=random.randint(10, Option_screen_width // 2),
        h=random.randint(10, Option_screen_height // 2),
        dx=random.uniform(Option_gen_speed_x[0], Option_gen_speed_x[1]),
        dy=random.uniform(Option_gen_speed_y[0], Option_gen_speed_y[1]))

    start_time = time.time()
    last_update_time = time.time()

    while True:
        game_settings.randomize(target)
        current_time = time.time()

        if current_time - last_update_time > 1:
            last_update_time = current_time

        target.move()
        target.randomize_size()
        target.randomize_position()
        target.randomize_velocity()

        if Option_gen_visualise:
            vision.queue.put(target)

        x, y = target.adjust_mouse_movement(
            target_x=target.x, target_y=target.y, game_settings=game_settings)

        data.add_target_data((game_settings.screen_width,
                              game_settings.screen_height,
                              game_settings.screen_x_center,
                              game_settings.screen_y_center,
                              game_settings.mouse_dpi,
                              game_settings.mouse_sensitivity,
                              game_settings.fov_x,
                              game_settings.fov_y,
                              target.x,
                              target.y,
                              x,
                              y))
        pbar.n = int(last_update_time - start_time)
        pbar.refresh()

        if int(last_update_time - start_time) >= Option_gen_time:
            if Option_gen_visualise:
                vision.queue.put(None)  # call break
            data.stop()
            pbar.close()
            break


if __name__ == "__main__":
    ###################### Options ######################

    # Game settings
    Option_screen_width = 380
    Option_screen_height = 400
    Option_fov_x = 40
    Option_fov_y = 25
    Option_mouse_dpi = 800
    Option_mouse_sensitivity = 2

    # Data
    Option_delete_prev_data = True
    
    # Generation settings
    Option_Generation = True
    Option_gen_time = 180
    Option_gen_visualise = True
    Option_gen_visualise_draw_line = True

    # Train
    Option_train = True
    Option_train_epochs = 40
    Option_train_batch_size = 4096
    Option_save_every_N_epoch = 5
    Option_learning_rate = 0.005

    # Speed - 1 is max
    Option_gen_speed_x = [-1, 1]
    Option_gen_speed_y = [-1, 1]

    # Game settings - random options
    Option_random_screen_resolution = False
    Option_random_screen_resolution_width = [150, 600]
    Option_random_screen_resolution_height = [150, 600]

    Option_random_fov = False
    Option_random_fov_x = [45, 60]
    Option_random_fov_y = [45, 45]

    Option_random_mouse_dpi = False
    Option_random_mouse_dpi_min_max = [800, 1000]

    Option_random_mouse_sensitivity = False
    Option_random_mouse_sensitivity_min_max = [1.05, 1.2]

    # Testing model
    Option_test_model = False

    #####################################################

    game_settings = Game_settings()

    data = Data(delete_prev_data=Option_delete_prev_data)

    if Option_gen_visualise:
        vision = Visualisation()

    if Option_train or Option_test_model:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if Option_Generation:
        gen_data()

    if Option_gen_visualise:
        vision.stop()

    if Option_train:
        train_net()

    if Option_test_model:
        test_net()
        data.stop()
    else:
        data.stop()
        vision.stop()
        gen_data.stop()
        train_net.stop()
        test_net.stop()
        