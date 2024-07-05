import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from config import Option_train_batch_size, Option_learning_rate, Option_train_epochs, Option_save_every_N_epoch, Option_gen_time, Option_screen_width, Option_screen_height, Option_gen_speed_x, Option_gen_speed_y, Option_gen_visualise, Option_mouse_dpi, Option_mouse_sensitivity, Option_fov_x, Option_fov_y
from data.dataset import CustomDataset
from models import Mouse_net
from data.data import data
from data.visualisation import visualisation
from utils.game_settings import game_settings
from utils.target import Target

def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    sec = seconds % 60

    formatted_time = f"{hours:02}:{minutes:02}:{sec:06.3f}"
    return formatted_time



def train_net():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    save_path = 'runs/'
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
    
       # Remove previous optimizer state
    optimizer_path = os.path.join(save_path, 'mouse_net_optimizer.pth')
    if os.path.exists(optimizer_path):
        os.remove(optimizer_path)
        print("Previous optimizer state removed.")

    # Now, load the model only if it exists (optimizer state is gone)
    model_path = os.path.join(save_path, 'mouse_net.pth')
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print("Loaded existing model state.")

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
            torch.save(model.state_dict(), os.path.join(save_path, f'mouse_net_epoch_{epoch + 1}.pth'))
            torch.save(optimizer.state_dict(), os.path.join(save_path, 'mouse_net_optimizer.pth'))  # Save optimizer state
            print(f'Model saved at epoch {epoch + 1}')
        
        if epoch in [10, 15, 20, 23, 25, 27, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]:
           lr = optimizer.param_groups[0]['lr']
           if lr > .0000001:
            lr = lr / 10
            # Update optimizer learning rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr  
            print(f"Learning rate reduced to: {lr}")

    plt.plot(loss_values)
    plt.title('Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

    torch.save(model.state_dict(), 'mouse_net.pth')

def test_net():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
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
    target = Target(
        x=random.randint(0, Option_screen_width - 4),  
        y=random.randint(0, Option_screen_height - 4), 
        w=random.randint(4, Option_screen_width),
        h=random.randint(4, Option_screen_height),
        dx=random.uniform(Option_gen_speed_x[0], Option_gen_speed_x[1]),
        dy=random.uniform(Option_gen_speed_y[0], Option_gen_speed_y[1]))

    start_time = time.time()
    last_update_time = time.time()
    prev_time = None
    prev_x = None
    prev_y = None

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
           visualisation.queue.put(Target(predicted_x, predicted_y, target.w, target.h, target.dx, target.dy))

        # Correct arguments for `adjust_mouse_movement` 
        x, y = target.adjust_mouse_movement(
        target_x=target.get_center()[0],  # Pass center_x
        target_y=target.get_center()[1],  # Pass center_y
        game_settings_module=game_settings)
        
        data.add_target_data((Option_screen_width,
                            Option_screen_height,
                            Option_screen_width // 2,
                            Option_screen_height // 2,
                            Option_mouse_dpi,
                            Option_mouse_sensitivity,
                            Option_fov_x,
                            Option_fov_y,
                            target.x,
                            target.y,
                            x,
                            y))

        # prediction
        if prev_time is not None:
            delta_time = current_time - prev_time
            if delta_time > 0:
                velocity_x = (target.x - prev_x) / delta_time
                velocity_y = (target.y - prev_y) / delta_time
                predicted_x = target.x + velocity_x * delta_time
                predicted_y = target.y + velocity_y * delta_time
            else:
                predicted_x = target.x
                predicted_y = target.y
        else:
            predicted_x = target.x
            predicted_y = target.y

        prev_x = target.x
        prev_y = target.y
        prev_time = current_time

        pbar.n = int(last_update_time - start_time)
        pbar.refresh()
        pbar.update(1)

        if int(last_update_time - start_time) >= Option_gen_time:
            if Option_gen_visualise:
                visualisation.queue.put(None)  # call break
            data.stop()
            pbar.close()
            break
