import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchsummary import summary

class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        self.block = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, x):
        return self.block(x)


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.block = nn.Sequential(
            #nn.Dropout2d(0.2),
            nn.Conv2d(1, 16, kernel_size=(3,3),padding=1),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=(3,3),padding=1),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=(3,3),padding=1),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.AvgPool2d(kernel_size=(2,2))
            #nn.Linear(512,2)
        )
        self.out = nn.Sequential(
            nn.Linear(256,1),
            nn.Sigmoid()
        )


    def forward(self, x):
        x = self.block(x)
        x = x.view(-1,x.size(1))
        #print(x.size())
        return self.out(x)


def dataloaders(name):

    os.makedirs('./data/mnist', exist_ok=True)
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Resize(image_size),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                       ])),
        batch_size=batch_size, shuffle=True, drop_last=True)

    return dataloader

def generate_images(gen_model, epoch_no, batch_size=60):

    gen_inp = Variable(torch.Tensor(np.random.normal(0, 1, (batch_size, 100, 1, 1))), requires_grad=False)
    #print(gen_inp[0:5])

    gen_inp = gen_inp.to(device)
    fake_images = gen_model(gen_inp)

    save_image(fake_images, "./saved_data/gen_mnist_img{}.png".format(epoch_no), nrow=6, padding=2, normalize=True)


if __name__ == "__main__":

    image_size = 64
    batch_size = 1024
    n_epochs = 200

    gen_lr = 1e-4
    dis_lr = 1e-4

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    gen_model = Generator().to(device)
    dis_model = Discriminator().to(device)

    optimizer_gen = optim.Adam(gen_model.parameters(), lr=gen_lr)
    optimizer_dis = optim.Adam(dis_model.parameters(), lr=dis_lr)

    #print(summary(gen_model, input_size=(100,1,1)))
    print(summary(dis_model, input_size=(1,64,64)))
    #exit(0)

    g_labels = Variable(torch.Tensor(batch_size, 1).fill_(1.0), requires_grad=False).to(device)
    d_true_labels = Variable(torch.Tensor(batch_size, 1).fill_(0.0), requires_grad=False).to(device)

    place_holder_false = np.ones([batch_size, 1])
    place_holder_true = np.zeros([batch_size, 1])

    loss = nn.BCELoss()

    loss_list = []

    disc_dataloader = dataloaders("custom")

    for epoch in range(n_epochs):
        print("Epoch {}".format(epoch))

        total = len(disc_dataloader) * batch_size * 2
        #total = 10000 * batch_size * 2
        correct_pos = 0.0
        correct_neg = 0.0
        gen_acc = 0.0

        disc_loss_epoch = 0
        gen_loss_epoch = 0

        for batch_id, (true_images,_) in enumerate(disc_dataloader, 0):
            # TRAINING GENERATOR

            optimizer_gen.zero_grad()

            # CREATING A BATCH OF RANDOM NOISE IN [0,1] FOR GENERATOR INPUT
            gen_inp = Variable(torch.Tensor(np.random.normal(0, 1, (batch_size, 100, 1, 1))),requires_grad=False)
            #print(gen_inp[0:5])

            gen_inp = gen_inp.to(device)
            fake_images = gen_model(gen_inp)

            #save_image(fake_images[0:32], "img{}.png".format(batch_id%5), nrow=8, padding=2, normalize=True)

            #print(fake_images.size())
            gen_loss = loss(dis_model(fake_images), d_true_labels)
            gen_loss.backward()
            optimizer_gen.step()


            # TRAINING DISCRIMINATOR
            optimizer_dis.zero_grad()

            true_images = true_images.to(device)

            d_true_loss=torch.mean(loss(dis_model(true_images), d_true_labels))
            d_fake_loss=torch.mean(loss(dis_model(fake_images.detach()), g_labels)) # Have to detach from graph for some reason

            d_loss = (d_true_loss+d_fake_loss)/2

            if d_loss.item()>0.6:
                print("discriminator updated")
                d_loss.backward()
                optimizer_dis.step()

            gen_loss_epoch+=gen_loss.item()
            disc_loss_epoch+=d_loss.item()
            print(gen_loss, d_loss)


            with torch.no_grad():
                outputs = dis_model(true_images)
                #_, predicted = torch.max(outputs.data, 1)
                #print(outputs)
                predicted = (outputs > 0.5).to("cpu").numpy()
                correct_pos += (predicted == place_holder_true).sum()

                outputs = dis_model(fake_images.detach())
                #_, predicted = torch.max(outputs.data, 1)
                predicted = (outputs > 0.5).to("cpu").numpy()
                print((predicted == place_holder_false).sum())
                correct_neg += (predicted == place_holder_false).sum()

                outputs = dis_model(fake_images.detach())
                # _, predicted = torch.max(outputs.data, 1)
                predicted = (outputs > 0.5).to("cpu").numpy()
                batch_gen_acc = (predicted == place_holder_true).sum()
                gen_acc += (predicted == place_holder_true).sum()

            # if batch_gen_acc/batch_size>0.5:
            #     print("discriminator updated")
            #     d_loss.backward()
            #     optimizer_dis.step()
            

        correct_pos/=total
        correct_neg/=total
        gen_acc/=(total / 2)

        print("Current Discriminator Accuracy = {}".format(correct_pos+correct_neg))
        print("Current Discriminator Positives Accuracy = {}".format(correct_pos*2))
        print("Current Discriminator Negatives Accuracy = {}".format(correct_neg*2))
        print("Current Generator Accuracy = {}".format(gen_acc))

        loss_list.append((gen_loss_epoch/len(disc_dataloader),disc_loss_epoch/len(disc_dataloader)))
        import pickle
        with open("./saved_data/loss_epoch_mnist", "wb") as f:
            pickle.dump(loss_list, f)

        if epoch%5==0:
            generate_images(gen_model, epoch, 36)


        torch.save(gen_model.state_dict(), "./saved_data/generator_mnist")
        torch.save(dis_model.state_dict(), "./saved_data/discriminator_mnist")
