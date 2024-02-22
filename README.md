# Mouse Movement Prediction Training and Visualization Tool

This project provides a Python script to simulate mouse movement, train a neural network on the generated data, and visualize the training process. The script uses `PyTorch` for machine learning and `OpenCV` for visualization.

## Description

The script is designed to generate artificial mouse movement data, train a simple neural network on this data to predict mouse movements, and optionally visualize the target movement and data generation process in real-time for [ai aimbot](https://github.com/SunOner/yolov8_aimbot).

### Features

- Generates mouse movement data with targets moving across the screen.
- Configurable game settings for screen resolution, field of view, mouse DPI, and sensitivity.
- Supports real-time visualization of the target movement using OpenCV.
- Provides options for randomizing game settings to diversify the training data.
- Writes generated data to a file for later use in training.
- Trains a simple feed-forward neural network to predict mouse movement.
- Allows model testing by making predictions on random data points.

### Usage

1. Configure the script options in the __main__ section of the script.
2. Run the script using Python to start the data generation and training process ```python train.py```.
3. If visualization is enabled (Option_visualise = True), an OpenCV window will display the target movements.
4. The trained model will be saved as mouse_net.pth.
5. Move mouse_net.pth to ai aimbot main folder.
6. Go to ai aimbot config and set AI_mouse_net = True.

### Configuration Options

- Option_visualise: Set to True to enable real-time visualization.

- Option_gen_max_targets, Option_gen_min_live_time, Option_gen_max_live_time: Configures target generation parameters.

- Option_screen_width, Option_screen_height, Option_fov_x, Option_fov_y, Option_mouse_dpi, Option_mouse_sensitivity: Defines game settings for the simulation.

- Randomization options (Option_random_screen_resolution, Option_random_fov, etc.): Enable variability in the simulation settings.

- Option_train_epochs: Number of epochs to train the neural network.

- Option_delete_prev_data: If set to True, previous data files will be deleted before starting a new session.

### Testing

- After training, the script automatically tests the model with a random data line from the generated file and prints out the predicted mouse movement.

### Contributing

- Contributions are welcome! Please feel free to submit a pull request or create an issue for any bugs or improvements.

# License

- This project is open-source and available under the MIT License.