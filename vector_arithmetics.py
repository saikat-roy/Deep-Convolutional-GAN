import torch
from dcgan import Generator
from PIL import Image
from torchvision.utils import save_image
from torchvision.transforms import ToPILImage
import numpy as np
import pickle

def make_vector_set(n_samples=3, vector_size=100):
    vector_set = torch.empty(n_samples,vector_size,1,1)
    counter = 0
    while counter<n_samples:
        gen_inp = torch.randn(1, vector_size, 1, 1, device=device, requires_grad=False)
        output = gen_model(gen_inp)  # .detach().numpy()
        # print(output)
        save_image(output, "1.png")
        img = None
        with open("1.png", "rb") as f:
            img = Image.open(f)
            img.show()
        i = input("Accept Image? 1: Accept, anything else: reject - ")
        if i == "1":
            print("Vector accepted")
            vector_set[counter,:]=gen_inp
            #print(vector_set)
            counter+=1
        else:
            img.close()

    return vector_set

if __name__ == "__main__":
    device = torch.device( 'cpu')
    model_save_path = r"./generator_mnist"
    gen_model = Generator()
    gen_model.load_state_dict(torch.load(model_save_path))

    n_samples = 3
    print("Selecting Vector Set 1")
    vector_set1 = make_vector_set(n_samples=n_samples)
    print("Selecting Vector Set 2")
    vector_set2 = make_vector_set(n_samples=n_samples)
    with open("saved_vectors.pkl", "wb") as f:
        pickle.dump((vector_set1,vector_set2),f)

    with open("saved_vectors.pkl", "rb") as f:
        vector_set1, vector_set2 = pickle.load(f)

    print(vector_set1.size(), vector_set2.size())

    save_image(gen_model(vector_set1), "vset1images.png", nrow=n_samples,normalize=True)
    save_image(gen_model(vector_set2), "vset2images.png", nrow=n_samples,normalize=True)

    vector1 = torch.mean(vector_set1, dim=0, keepdim=True)
    vector2 = torch.mean(vector_set2, dim=0, keepdim=True)

    save_image(gen_model(vector1), "mean_img_1.png",normalize=True)
    save_image(gen_model(vector2), "_mean_img2.png",normalize=True)
    save_image(gen_model(vector1+vector2), "add_img.png",normalize=True)
    save_image(gen_model(vector1-vector2), "sub_img12.png",normalize=True)
    save_image(gen_model(vector2-vector1), "sub_img21.png",normalize=True)


