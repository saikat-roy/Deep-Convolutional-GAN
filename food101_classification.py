
import numpy as np
import torch
from torchvision import datasets

from dcgan import Discriminator

def penultimate_layer_features(model, dataloader):
    None


if __name__ == "__main__":
    model_save_path = r"./saved_data/ImageNet/discriminator_ImageNet"
    disc_model = Discriminator()
    disc_model.load_state_dict(torch.load(model_save_path))

    food_101_path = r"/home/data/food-101/images"

    index_arr = np.arange(1000)
    np.random.shuffle(index_arr)
    print(index_arr)
