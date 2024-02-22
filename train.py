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

class Data:
    def __init__(self, delete_prev_data=True):
        self.data_path = './data.txt'
        self.data_queue = queue.Queue()
        self.delete_data()
        self.thread = threading.Thread(target=self.write_to_file)
        self.running = True
        self.thread.start()
        
    def write_to_file(self):
        while self.running:
            target = self.data_queue.get()
            if target is None:
                break
            with open(self.data_path, 'a') as f:
                f.write(f'{target[0]} {target[1]} {target[2]} {target[3]} {target[4]} {target[5]} {target[6]} {target[7]} {target[8]} {target[9]} {target[10]} {target[11]}\n')
    
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
        self.data_queue.put(None)
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
    
    def randomize(self):
        if Option_random_screen_resolution:
            self.screen_width = random.randint(300, 700)
            self.screen_height = random.randint(300, 700)
        self.screen_x_center = int(self.screen_width / 2)
        self.screen_y_center = int(self.screen_height / 2)
        if Option_random_fov:
            self.fov_x = random.randint(55, 100)
            self.fov_y = random.randint(55, 100)
        if Option_random_mouse_dpi:
            self.mouse_dpi = random.randint(500, 2000)
        if Option_random_mouse_sensitivity:
            self.mouse_sensitivity = random.uniform(1, 5)
        
class Target:
    def __init__(self, x, y, w, h, live_time, dx, dy):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.live_time = live_time
        self.dx = dx
        self.dy = dy
    
    def move(self):
        self.x += self.dx
        self.y += self.dy
        
        if self.x + self.w > game_settings.screen_width:
            self.x = game_settings.screen_width - self.w
            self.dx = -self.dx
        
        if self.x < 0:
            self.x = 0
            self.dx = -self.dx
        
        if self.y + self.h > game_settings.screen_height:
            self.y = game_settings.screen_height - self.h
            self.dy = -self.dy
        
        if self.y < 0:
            self.y = 0
            self.dy = -self.dy
    
    def update_velocity(self):
        self.dx += random.uniform(-0.5, 0.5)
        self.dy += random.uniform(-0.5, 0.5)

        self.dx = max(-1, min(self.dx, 1))
        self.dy = max(-1, min(self.dy, 1))
        
class Visualisation(threading.Thread):
    def __init__(self, data_obj):
        super(Visualisation, self).__init__()
        self.queue = queue.Queue(maxsize=999)
        self.cv2_window_name = 'train_mouse_net'
        self.data_obj = data_obj
        self.start()
    
    def run(self):
        while True:
            data = self.queue.get()
            if data is None:
                break

            image = np.zeros((game_settings.screen_height, game_settings.screen_width, 3), np.uint8)
            cv2.rectangle(image, (int(data.x), int(data.y)), (int(data.x + data.w), int(data.y + data.h)), (0, 255, 0), 2)
            cv2.imshow(self.cv2_window_name, image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
def adjust_mouse_movement_proxy(target_x, target_y):
        offset_x = target_x - game_settings.screen_x_center
        offset_y = target_y - game_settings.screen_y_center

        degrees_per_pixel_x = game_settings.fov_x / game_settings.screen_width
        degrees_per_pixel_y = game_settings.fov_y / game_settings.screen_height
        
        mouse_move_x = offset_x * degrees_per_pixel_x

        mouse_dpi_move_x = (mouse_move_x / 360) * (game_settings.mouse_dpi * (1 / game_settings.mouse_sensitivity))

        mouse_move_y = offset_y * degrees_per_pixel_y
        mouse_dpi_move_y = (mouse_move_y / 360) * (game_settings.mouse_dpi * (1 / game_settings.mouse_sensitivity))

        return mouse_dpi_move_x, mouse_dpi_move_y
    
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

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(10, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
def train_net():
    print(f'Starting train mouse_net model.\nUsing device: {device}.')
    dataset = CustomDataset(data.data_path)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    model = SimpleNN().to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = Option_train_epochs
    for epoch in range(epochs):
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

    torch.save(model.state_dict(), 'mouse_net.pth')
    
def test_model(input_data):
    model = SimpleNN().to(device)
    model.load_state_dict(torch.load('mouse_net.pth', map_location=device))
    model.eval()

    input_tensor = torch.tensor(input_data, dtype=torch.float32).to(device)

    with torch.no_grad():
        prediction = model(input_tensor)

    return prediction.cpu().numpy()

def read_random_line():
    with open(data.data_path, 'r') as file:
        lines = file.readlines()
        random_line = random.choice(lines).strip()
        return random_line

def convert_to_float_list(line):
    return [float(number) for number in line.split()]

def gen_data():
    targets = [
    Target(
        x=random.randint(0, game_settings.screen_width),
        y=random.randint(0, game_settings.screen_height),
        w=random.uniform(1, 400),
        h=random.uniform(1, 400),
        live_time=random.randint(Option_gen_min_live_time, Option_gen_max_live_time),
        dx=random.uniform(Option_gen_min_speed_x, Option_gen_max_speed_x),
        dy=random.uniform(Option_gen_min_speed_y, Option_gen_max_speed_y)) for _ in range(Option_gen_max_targets)]

    start_time = time.time()
    last_update_time = time.time()
    
    pbar = tqdm(total=Option_gen_max_targets, desc='Targets')
    while data.running:
        current_time = time.time()
        alive_targets = []

        if current_time - last_update_time > 1:
            for target in targets:
                target.update_velocity()
            last_update_time = current_time

        for target in targets:
            if current_time - start_time < target.live_time:
                alive_targets.append(target)
                target.move()
                
                if Option_visualise:
                    vision.queue.put(target)
                
                move_proxy = adjust_mouse_movement_proxy(target_x=target.x, target_y=target.y)
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
                                      move_proxy[0],
                                      move_proxy[1]))
                game_settings.randomize()
        targets = alive_targets
        pbar.n = len(targets)
        pbar.refresh()
        
        if len(targets) == 0:
            data.stop()
            pbar.close()
            break
        
    if Option_visualise:
        vision.queue.put(None)
        vision.join()

if __name__ == "__main__":
    ###################### Options ######################
    
    # Visual
    Option_visualise = False
    
    # Generation
    Option_gen_max_targets = 100
    Option_gen_min_live_time = 60
    Option_gen_max_live_time = 120
    Option_gen_min_speed_x = -10
    Option_gen_max_speed_x = 10
    Option_gen_min_speed_y = -10
    Option_gen_max_speed_y = 10
    
    # Game settings
    Option_screen_width = 580
    Option_screen_height = 420
    Option_fov_x = 90
    Option_fov_y = 55
    Option_mouse_dpi = 1000
    Option_mouse_sensitivity = 1.2
    
    # Game settings - random options
    Option_random_screen_resolution = False
    Option_random_fov = False
    Option_random_mouse_dpi = False
    Option_random_mouse_sensitivity = False
    
    # Train
    Option_train_epochs = 5
    
    # Data
    Option_delete_prev_data = True
    #####################################################

    game_settings = Game_settings()
    data = Data(Option_delete_prev_data)
    if Option_visualise:
        vision = Visualisation(data)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    gen_data()
    
    train_net()
    
    # Test model
    print('Starting testing model.')
    random_line = read_random_line()
    data_list = convert_to_float_list(random_line)
    input_data = data_list[:10]
    output = test_model(input_data)
    print(output)