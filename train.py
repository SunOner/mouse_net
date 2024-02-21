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
            except Exception as e:
                print(f"Error deleting data file: {e}")
            try:
                os.remove('mouse_net.pth')
            except Exception as e:
                print(f"Error deleting data file: {e}")

    def add_target_data(self, target):
        self.data_queue.put(target)
        
    def stop(self):
        self.running = False
        self.data_queue.put(None)
        self.thread.join()

class Game_settings:
    def __init__(self):
        self.screen_width = 580
        self.screen_height = 420
        self.screen_x_center = int(self.screen_width / 2)
        self.screen_y_center = int(self.screen_height / 2)
        self.fov_x = 90
        self.fov_y = 55
        self.mouse_dpi = 1000
        self.mouse_sensitivity = 1.2
    
    def randomize(self):
        self.screen_width = random.randint(50, 920)
        self.screen_height = random.randint(50, 920)
        self.screen_x_center = int(self.screen_width / 2)
        self.screen_y_center = int(self.screen_height / 2)
        self.fov_x = random.randint(10, 130)
        self.fov_y = random.randint(10, 130)
        self.mouse_dpi = random.randint(100, 5000)
        self.mouse_sensitivity = random.uniform(0.1, 20.0)
        
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
        self.x = max(0, min(self.x, game_settings.screen_width - self.w))
        self.y = max(0, min(self.y, game_settings.screen_height - self.h))
    
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
        w=random.uniform(1, 180),
        h=random.uniform(1, 180),
        live_time=random.randint(Option_gen_min_live_time, Option_gen_max_live_time),
        dx=random.uniform(-1, 1),
        dy=random.uniform(-1, 1)) for _ in range(Option_gen_max_targets)]

    update_interval = 1
    start_time = time.time()
    last_update_time = time.time()
    
    pbar = tqdm(total=Option_gen_max_targets, desc='Generation data')
    while data.running:
        current_time = time.time()
        alive_targets = []

        if current_time - last_update_time > update_interval:
            for target in targets:
                target.update_velocity()
            last_update_time = current_time

        for target in targets:
            if current_time - start_time < target.live_time:
                game_settings.randomize()
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
    Option_visualise = False
    Option_gen_max_targets = 10
    Option_gen_min_live_time = 50
    Option_gen_max_live_time = 100
    Option_train_epochs = 5
    Option_delete_prev_data = False
    # Init classes
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