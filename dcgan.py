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

from utils import CustomFaces
# class GeneratorB(nn.Module):
#
#     def __init__(self):
#         super(GeneratorB, self).__init__()
#         self.block = nn.Sequential(
#             # input is Z, going into a convolution
#             nn.ConvTranspose2d(100, 256, 4, 1, 0, bias=False),
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(0.2,True),
#             # state size. (ngf*8) x 4 x 4
#             nn.ConvTranspose2d(256, 256, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(0.2,True),
#             # state size. (ngf*4) x 8 x 8
#             nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.2,True),
#             # state size. (ngf*2) x 16 x 16
#             nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(0.2,True),
#             # state size. (ngf) x 32 x 32
#             nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),
#             nn.Tanh()
#             # state size. (nc) x 64 x 64
#         )
#
#     def forward(self, x):
#         return self.block(x)
#
#
# class DiscriminatorB(nn.Module):
#
#     def __init__(self):
#         super(DiscriminatorB, self).__init__()
#         self.block = nn.Sequential(
#             #nn.Dropout2d(0.2),
#             nn.Conv2d(1, 16, kernel_size=(3,3),padding=1, bias=False),
#             nn.BatchNorm2d(16),
#             nn.LeakyReLU(0.2,inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             #nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(16, 32, kernel_size=(3, 3), padding=1, bias=False),
#             nn.BatchNorm2d(32),
#             nn.LeakyReLU(0.2,inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(32, 64, kernel_size=(3,3),padding=1, bias=False),
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(0.2,inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1, bias=False),
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(128, 1, kernel_size=(3,3),padding=1, bias=False),
#             #nn.BatchNorm2d(1),
#             #nn.LeakyReLU(inplace=True),
#             #nn.MaxPool2d(kernel_size=2,stride=2),
#             nn.AvgPool2d(kernel_size=(4,4)),
#             nn.Sigmoid()
#             #nn.Linear(512,2)
#         )
#
#
#     def forward(self, x):
#         x = self.block(x)
#         x = x.view(-1)
#         #print(x.size())
#         return x
#

nc = 3
ndf = 64
ngf = 128
nz = 100

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
                    # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
                    # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
                    # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
                    # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
                    # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
                    # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
                    # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
                    # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
                    # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
                    # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
                    # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        x = self.main(input)
        return x.view(-1)



def dataloaders(name):
    transforms_list = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    if name == "mnist":
        os.makedirs('./data/mnist', exist_ok=True)
        dataset = datasets.MNIST('./data/mnist', train=False, download=True,
                       transform=transforms_list)
    elif name == "custom_faces":
        dataset = CustomFaces(r"/home/saikat/PycharmProjects/DCGAN/data/custom_face", transform=transforms_list)
    elif name == "LSUN":
        #os.makedirs('./data/lsun', exist_ok=True)
        dataset = datasets.LSUN(r"/home/data/LSUN" ,["church_outdoor_train"], transforms_list)
    elif name == "imagenet":
        return NotImplementedError

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)




    return dataloader

def generate_images(gen_model, epoch_no, batch_size=60):
    #gen_model.eval()
    #gen_inp = Variable(torch.Tensor(np.random.normal(0, 1, (batch_size, 100, 1, 1))), requires_grad=False)
    gen_inp = torch.randn(batch_size, 100, 1, 1, device=device)
    #print(gen_inp[0:5])

    gen_inp = gen_inp.to(device)
    fake_images = gen_model(gen_inp)

    save_image(fake_images, "./saved_data/gen_mnist_img{}.png".format(epoch_no), nrow=10, normalize=False)
    #fake_images = fake_images.detach().cpu()
    #save_image(fake_images, "./saved_data/gen_mnist_img{}_method2.png".format(epoch_no), nrow=6, padding=2, normalize=False)



def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


if __name__ == "__main__":

    image_size = 64
    batch_size = 128
    n_epochs = 150

    gen_lr = 1e-3
    dis_lr = 1e-4

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    gen_model = Generator().to(device)
    dis_model = Discriminator().to(device)

    gen_model.apply(weights_init)
    dis_model.apply(weights_init)

    optimizer_gen = optim.Adam(gen_model.parameters(), lr=gen_lr, betas=(0.5, 0.999))
    optimizer_dis = optim.Adam(dis_model.parameters(), lr=dis_lr, betas=(0.5, 0.999))

    print(summary(gen_model, input_size=(100,1,1)))
    print(summary(dis_model, input_size=(3,64,64)))
    #exit(0)

    true_labels = 0.0   # CHANGE TO ZERO TO FLIP LABELS

    d_true_labels = Variable(torch.Tensor(batch_size).fill_(true_labels), requires_grad=False).to(device)
    g_labels = Variable(torch.Tensor(batch_size).fill_(1.0-true_labels), requires_grad=False).to(device)

    place_holder_true = np.ones([batch_size]) * true_labels
    place_holder_false = np.ones([batch_size]) * (1.0-true_labels)

    loss = nn.BCELoss().to(device)

    loss_list = []

    disc_dataloader = dataloaders("LSUN")

    for epoch in range(n_epochs):
        print("Epoch {}".format(epoch))

        total = len(disc_dataloader) * batch_size * 2
        #total = 10000 * batch_size * 2
        correct_pos = 0.0
        correct_neg = 0.0
        gen_acc = 0.0

        disc_loss_epoch = 0
        gen_loss_epoch = 0

        #for batch_id, (true_images,_) in enumerate(disc_dataloader, 0): # For MNIST Example
        for batch_id, (true_images,_) in enumerate(disc_dataloader, 0):
            gen_model.train()
            dis_model.train()

            optimizer_dis.zero_grad()
            gen_inp_1 = torch.randn(batch_size, 100, 1, 1, device=device, requires_grad=False)

            # gen_inp = gen_inp.to(device)
            fake_images = gen_model(gen_inp_1)
            # TRAINING DISCRIMINATOR

            true_images = true_images.to(device)
            # print(loss(dis_model(true_images), d_true_labels))
            # print(loss(dis_model(fake_images.detach()), g_labels))
            d_true_loss = loss(dis_model(true_images), d_true_labels)
            d_fake_loss = loss(dis_model(fake_images.detach()), g_labels)  # Have to detach from graph for some reason

            d_loss = (d_true_loss + d_fake_loss) / 2

            # if d_loss.item()>0.6:
            #    print("discriminator updated")
            d_true_loss.backward()
            d_fake_loss.backward()
            optimizer_dis.step()
            # if g_loss.item()>0.75


            # TRAINING GENERATOR

            optimizer_gen.zero_grad()

            # CREATING A BATCH OF RANDOM NOISE IN [0,1] FOR GENERATOR INPUT
            #gen_inp = Variable(torch.Tensor(np.random.normal(0, 1, (batch_size, 100, 1, 1))),requires_grad=False)
            #print(gen_inp[0:5])

            #save_image(fake_images[0:32], "img{}.png".format(batch_id%5), nrow=8, padding=2, normalize=True)

            #print(fake_images.size())
            gen_loss = loss(dis_model(fake_images), d_true_labels)
            gen_loss.backward()
            optimizer_gen.step()

            gen_loss_epoch += gen_loss.item()
            disc_loss_epoch += d_loss.item()
            #print(gen_loss.item(), d_loss.item())

            with torch.no_grad():
                outputs = dis_model(true_images)
                #_, predicted = torch.max(outputs.data, 1)
                #print(outputs)
                predicted = (outputs > 0.5).to("cpu").numpy()
                batch_disc_acc = (predicted == place_holder_true).sum()/batch_size
                correct_pos += (predicted == place_holder_true).sum()

                outputs = dis_model(fake_images.detach())
                #print(outputs)
                #_, predicted = torch.max(outputs.data, 1)
                predicted = (outputs > 0.5).to("cpu").numpy()
                #print(predicted, place_holder_false)
                #print((predicted == place_holder_false).sum())
                correct_neg += (predicted == place_holder_false).sum()

                outputs = dis_model(fake_images.detach())
                # _, predicted = torch.max(outputs.data, 1)
                predicted = (outputs > 0.5).to("cpu").numpy()
                batch_gen_acc = (predicted == place_holder_true).sum()/batch_size
                gen_acc += (predicted == place_holder_true).sum()

            #print(batch_gen_acc, batch_disc_acc+(1-batch_gen_acc))
            #if batch_disc_acc+(1-batch_gen_acc)<0.4:
            #     print("discriminator updated")
            #     d_loss.backward()
            #     optimizer_dis.step()
            #if batch_gen_acc<0.75:
            #    print("generator updated")
            #    gen_loss.backward()
            #    optimizer_gen.step()


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

        print(gen_loss_epoch / len(disc_dataloader), disc_loss_epoch / len(disc_dataloader))

        #if epoch%5==0:
        generate_images(gen_model, epoch, 100)


        torch.save(gen_model.state_dict(), "./saved_data/generator_mnist")
        torch.save(dis_model.state_dict(), "./saved_data/discriminator_mnist")
