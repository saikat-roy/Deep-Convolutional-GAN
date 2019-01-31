import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchsummary import summary

from dcgan import Discriminator

from sklearn.svm import SVC

nc = 3
ndf = 64
ngf = 128
image_size = 64
batch_size = 128

class Discriminator_trimmed(nn.Module):

    def __init__(self,list_children):
        super(Discriminator_trimmed, self).__init__()
        self.main = nn.Sequential(*list_children[0:-2])

    def forward(self, input):
        x = self.main(input)
        return x.view(-1,512*4*4)


if __name__ == "__main__":
    model_save_path = r"./saved_data/ImageNet/discriminator_ImageNet"
    disc_model = Discriminator()
    disc_model.load_state_dict(torch.load(model_save_path))
    list_children = []
    for i in disc_model.children():
        list_children.extend([*i])
    print(list_children)
    disc_new = Discriminator_trimmed(list_children)

    food_101_path = r"/home/data/food-101/images"

    index_arr = np.arange(1000)
    np.random.shuffle(index_arr)
    print(index_arr)
    training_idx = index_arr[0:750]
    validation_idx = index_arr[750:]

    for i in range(1,101):
        training_idx_2=index_arr[0:750]+(i*1000)
        training_idx = np.append(training_idx, training_idx_2)

        validation_idx_2 = index_arr[750:1000]+(i*1000)
        validation_idx = np.append(validation_idx, validation_idx_2)

    train_x = np.empty((training_idx.shape[0], 512 * 4 * 4))
    valid_x = np.empty((validation_idx.shape[0], 512 * 4 * 4))
    train_y = np.empty((training_idx.shape[0]))
    valid_y = np.empty((validation_idx.shape[0]))

    print(training_idx.shape[0])
    print(validation_idx.shape[0])

    train_sampler = torch.utils.data.SubsetRandomSampler(training_idx)
    valid_sampler = torch.utils.data.SubsetRandomSampler(validation_idx)

    transforms_list = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = datasets.ImageFolder(food_101_path, transforms_list)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                    sampler=valid_sampler)

    print(len(train_loader))
    print(len(validation_loader))

    disc_new = disc_new.to("cuda:0")
    print(summary(disc_new, input_size=(3, 64, 64)))

    disc_new.eval()
    i = 0
    for epoch_id, (X,y) in enumerate(train_loader):
        X = X.to("cuda:0")
        i+=X.size(0)
        train_x[(epoch_id)*X.size(0):(epoch_id+1)*X.size(0),:] = disc_new(X).detach().cpu().numpy()
        train_y[(epoch_id) * X.size(0):(epoch_id+1) * X.size(0)] = y.numpy()
        #print(train_y[(epoch_id) * X.size(0):(epoch_id+1) * X.size(0)])
    print(i)
    for epoch_id, (X, y) in enumerate(validation_loader):
        X = X.to("cuda:0")
        valid_x[(epoch_id) * X.size(0):(epoch_id+1) * X.size(0), :] = disc_new(X).detach().cpu().numpy()
        valid_y[(epoch_id) * X.size(0):(epoch_id+1) * X.size(0)] = y.numpy()

    print(np.unique(train_y))
    clf = SVC(gamma='auto')
    clf.fit(train_x, train_y)
    print(clf.score(valid_x, valid_y))






