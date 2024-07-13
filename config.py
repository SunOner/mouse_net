# Game settings
# config.py
# Screen width in pixels
Option_screen_width = 350
# Screen height in pixels
Option_screen_height = 400
# Field of view in x direction (game settings) affects the mouse left and right
Option_fov_x = 70
# Field of view in y direction (game settings) affects the mouse up and down
Option_fov_y = 60
# Mouse DPI (dots per inch) - higher DPI means more sensitive virtual mouse
Option_mouse_dpi = 1000
# Mouse sensitivity means how much the mouse moves in relation to the virtual mouse DPI
Option_mouse_sensitivity = 5.0
# Data settings
# Delete previous generated data True ot False (use capital letter on first letter of True or False) default is True
Option_delete_prev_data = True
# Generation settings True ot False (use capital letter on first letter of True or False) default is True
Option_Generation =True
# Time in seconds to generate data
Option_gen_time = 180
# visualise generation True ot False (use capital letter on first letter of True or False) default is True
Option_gen_visualise = True
# Visualise draw line True ot False (use capital letter on first letter of True or False) default is True
Option_gen_visualise_draw_line = True
# Train model settings True ot False (use capital letter on first letter of True or False)  default is True
Option_train = True
# Number of epochs how many rounds to train the model
Option_train_epochs = 40 
# Batch size - how many samples to process at once (higher is faster but requires more memory)
Option_train_batch_size = 4096 
 # Save every N epoch - how may epochs to save the model periodically
Option_save_every_N_epoch = 10
# Learning rate The learning rate controls how quickly the model is adapted to the problem.
# Smaller values require more training epochs, but larger values may not train the model effectively.
Option_learning_rate = 0.01
# Speed - 1 is max speed, 0.5 is half speed, -1 is reverse max speed and -0.5 is reverse half speed  default is 1
Option_gen_speed_x = [-1, 1]
Option_gen_speed_y = [-1, 1]
# Game settings - random values
# Random screen resolution True ot False (use capital letter on first letter of True or False) default is False
Option_random_screen_resolution = False
 # Random screen resolution width (random screen size in pixels left and right)
Option_random_screen_resolution_width = [150, 600]
 # Random screen resolution height (random screen size in pixels up and down)
Option_random_screen_resolution_height = [150, 600]
# Random field of view True ot False (use capital letter on first letter of True or False)
Option_random_fov = False
 # Random field of view in x direction (random field of view in pixels left and right)
Option_random_fov_x = [45, 90]
 # Random field of view in y direction (random field of view in pixels up and down)
Option_random_fov_y = [45, 90]
# Random mouse  DPI True ot False (use capital letter on first letter of True or False)
Option_random_mouse_dpi = False
 # Random mouse DPI min and max (random mouse DPI in pixels)
Option_random_mouse_dpi_min_max = [1000, 3000]
# Random mouse sensitivity True ot False (use capital letter on first letter of True or False)
Option_random_mouse_sensitivity = False
 # Random mouse sensitivity min and max (random mouse sensitivity in pixels)
Option_random_mouse_sensitivity_min_max = [1.0, 3.0]
# Testing model settings True ot False (use capital letter on first letter of True or False)
Option_test_model = True