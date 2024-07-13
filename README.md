<div align="center">

# Mouse Movement Prediction Training Tool
[![Discord server](https://badgen.net/discord/online-members/sunone)](https://discord.gg/sunone)
  <p>
    <a href="https://github.com/SunOner/sunone_aimbot/" target="_blank">
      <img width="75%" src="https://raw.githubusercontent.com/SunOner/mouse_net/main/media/mouse_net.gif"></a>
  </p>
</div>

This project provides a Python script to simulate mouse movement, train a neural network on the generated data, and visualize the training process. The script uses `PyTorch` for machine learning and `OpenCV` for visualization.

## Description

The script is designed to generate artificial mouse movement data, train a simple neural network on this data to predict mouse movements, and optionally visualize the target movement and data generation process in real-time for [ai aimbot](https://github.com/SunOner/sunone_aimbot).

## Features

- Generates mouse movement data with targets moving across the screen.
- Configurable game settings for screen resolution, field of view, mouse DPI, and sensitivity.
- Supports real-time visualization of the target movement using OpenCV.
- Provides options for randomizing game settings to diversify the training data.
- Writes generated data to a file for later use in training.
- Trains a simple feed-forward neural network to predict mouse movement.
- Allows model testing by making predictions on random data points.

## Usage

1. Install the required libraries using `pip`: `pip install numpy opencv-python torch torchvision tqdm matplotlib scikit-learn`
2. Configure the script options in the config.py file.
3. Run the script using Python to start the data generation and training process `python main.py`.
4. The trained model will be saved as mouse_net.pth in main folder.
5. Move mouse_net.pth to Sunone Aimbot main folder.
6. Go to ai aimbot config and set AI_mouse_net = True.

## Configuration Options

The provided code is a Python script related to a machine learning project that involves training a neural network to predict mouse movements based on game data. The script includes class definitions, training and testing procedures, and a number of configurable options. Below are descriptions of the configurable options:

### Data Options
- `Option_delete_prev_data`: Determines whether or not to delete previously stored data files before starting the data generation process. If set to `True`, the existing data.txt and mouse_net.pth files will be removed.

### Training Options
- `Option_train`: Indicates whether to train the neural network model.
- `Option_train_epochs`: Specifies the number of epochs to train the model for.
- `Option_batch_size`: Sets the batch size for training the neural network.
- `Option_learning_rate`: Sets the learning rate for training the neural network.

### Testing Options
- `Option_test_model`: If set to `True`, the script will run model evaluation on a random sample from the generated data.

### Data Generation Options
- `Option_Generation`: Enables the generation of data when set to `True`.
- `Option_gen_time`: Defines the duration (in seconds) for which data generation will run.
- `Option_gen_visualise`: Controls whether to visualize the target's position on the screen as the data is collected.
- `Option_gen_visualise_draw_line`: If `True`, a line will be drawn from the center of the screen to the target's position during visualization.

### Size and Speed Options
- `Option_min_w` and `Option_max_w`: Define the minimum and maximum width of the target.
- `Option_min_h` and `Option_max_h`: Define the minimum and maximum height of the target.
- `Option_gen_min_speed_x` and `Option_gen_max_speed_x`: Specify the minimum and maximum horizontal velocities of the target.
- `Option_gen_min_speed_y` and `Option_gen_max_speed_y`: Specify the minimum and maximum vertical velocities of the target.

### Game Settings Options
- `Option_screen_width` and `Option_screen_height`: Define the resolution of the screen.
- `Option_fov_x` and `Option_fov_y`: Set the field of view angles in the x-axis and y-axis, respectively.
- `Option_mouse_dpi`: The mouse's dots per inch (DPI) sensitivity setting.
- `Option_mouse_sensitivity`: The in-game mouse sensitivity setting.

### Random Game Settings Options
- `Option_random_screen_resolution`: Enables randomization of screen resolution if set to `True`.
- `Option_random_min_screen_resolution_width` and `Option_random_max_screen_resolution_width`: The range within which the screen width can be randomized.
- `Option_random_min_screen_resolution_height` and `Option_random_max_screen_resolution_height`: The range within which the screen height can be randomized.
- `Option_random_fov`: Enables randomization of the field of view if set to `True`.
- `Option_random_min_fov_x` and `Option_random_max_fov_x`: The range within which the FOV in the x-axis can be randomized.
- `Option_random_min_fov_y` and `Option_random_max_fov_y`: The range within which the FOV in the y-axis can be randomized.
- `Option_random_mouse_dpi`: Enables randomization of mouse DPI if set to `True`.
- `Option_random_min_mouse_dpi` and `Option_random_max_mouse_dpi`: The range within which the mouse DPI can be randomized.
- `Option_random_mouse_sensitivity`: Enables randomization of mouse sensitivity if set to `True`.
- `Option_random_min_mouse_sensitivity` and `Option_random_max_mouse_sensitivity`: The range within which the mouse sensitivity can be randomized.

These options allow the user to customize the behavior of the script to match different scenarios and requirements. The training and testing outcomes can vary significantly based on how these options are configured.

## Testing

- After training, the script automatically tests the model with a random data line from the generated file and prints out the predicted mouse movement.

## Contributing

- Contributions are welcome! Please feel free to submit a pull request or create an issue for any bugs or improvements.

## License

- This project is open-source and available under the MIT License.