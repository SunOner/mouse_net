# main.py
from config import *
from data.data import data
from data.visualisation import visualisation
from train import train_net, test_net, gen_data

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

data.stop()