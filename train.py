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
from sklearn.metrics import r2_score

from config import *
from data.dataset import CustomDataset
from models import Mouse_net
from data.data import data
from data.visualisation import visualisation
from utils.game_settings import game_settings
from utils.target import Target

def format_time(seconds):
    """Format time in hours, minutes, and seconds."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    sec = seconds % 60
    return f"{hours:02}:{minutes:02}:{sec:06.3f}"

def train_net():
    """Train the neural network."""
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    save_path = 'runs/'
    os.makedirs(save_path, exist_ok=True)
    print(f'Starting train mouse_net model.\nUsing device: {device}.')
    
    dataset = CustomDataset(data.data_path)
    dataloader = DataLoader(dataset, batch_size=Option_train_batch_size, shuffle=True, pin_memory=True)
    
    model = Mouse_net().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=Option_learning_rate)

    def lr_lambda(epoch):
        return 1.0 if epoch < 2 else 0.9 ** (epoch - 2)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    epochs = Option_train_epochs
    loss_values = []

    best_loss = float('inf')
    patience = 4
    epochs_without_improvement = 0
    best_epoch = 0

    start_time = time.time()
    print(f'Initial learning rate: {Option_learning_rate}')

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
        print(f'Epoch {epoch + 1}/{epochs} Loss: {epoch_loss:.5f} {format_time(train_time)} Current LR: {scheduler.get_last_lr()[0]}')
        
        scheduler.step() 

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            epochs_without_improvement = 0
            best_epoch = epoch
            torch.save(model.state_dict(), 'best_mouse_net.pth')
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience and epoch >= best_epoch + patience:
                print(f'Early stopping at epoch {epoch + 1}.')
                break

        if (epoch + 1) % Option_save_every_N_epoch == 0:
            torch.save(model.state_dict(), os.path.join(save_path, f'mouse_net_epoch_{epoch + 1}.pth'))
            print(f'Model saved at epoch {epoch + 1}')

    plt.plot(loss_values)
    plt.title('Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

    torch.save(model.state_dict(), 'mouse_net.pth')
    print('Model saved.')

def test_net(model_path='mouse_net.pth', test_data_path='data.txt'):
    """Test the trained neural network."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Starting testing model...')

    test_dataset = CustomDataset(test_data_path)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = Mouse_net().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    predictions, actuals = [], []

    with torch.no_grad():
        for inputs, targets in test_dataloader:
            inputs = inputs.to(device)
            prediction = model(inputs)
            predictions.append(prediction.cpu().numpy())
            actuals.append(targets.cpu().numpy())

    predictions = np.vstack(predictions)
    actuals = np.vstack(actuals)

    mse = np.mean((predictions - actuals) ** 2)
    mae = np.mean(np.abs(predictions - actuals))
    var_actuals = np.var(actuals)
    r2 = 1 - (mse / var_actuals)

    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"R-squared (R²): {r2}")

    for i in range(actuals.shape[1]):
        plt.scatter(actuals[:, i], predictions[:, i], alpha=0.5)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title(f'Actual vs Predicted for Target {i + 1}')
    plt.plot([actuals[:, i].min(), actuals[:, i].max()], [actuals[:, i].min(), actuals[:, i].max()], 'r--')
    plt.show()

def gen_visualise():
    """Generate visualization."""
    plt.show()

def gen_data():
    """Generate data for training."""
    pbar = tqdm(total=Option_gen_time, desc='Data generation')

    target = Target(
        x=random.randint(0, Option_screen_width),
        y=random.randint(0, Option_screen_height),
        w=random.randint(4, Option_screen_width),
        h=random.randint(4, Option_screen_height),
        dx=random.uniform(Option_gen_speed_x[0], Option_gen_speed_x[1]),
        dy=random.uniform(Option_gen_speed_y[0], Option_gen_speed_y[1])
    )

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
            visualisation.queue.put(target)

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

        if Option_gen_visualise:
            visualisation.queue.put(Target(predicted_x, predicted_y, target.w, target.h, target.dx, target.dy))

        x, y = target.adjust_mouse_movement(
            target_x=predicted_x, target_y=predicted_y, game_settings=game_settings)

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
        pbar.n = int(last_update_time - start_time)
        pbar.refresh()

        if int(last_update_time - start_time) >= Option_gen_time:
            if Option_gen_visualise:
                visualisation.queue.put(None)
            data.stop()
            pbar.close()
            break

if __name__ == "__main__":
    if Option_Generation:
        gen_data()

    if Option_gen_visualise:
        visualisation.stop()

    if Option_train:
        train_net()

    if Option_test_model:
        test_net()

    data.stop()
