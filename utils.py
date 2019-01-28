import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class CustomD(Dataset):

    def __init__(self, folderpath, transform=None):
        """
        :param folderpath: Path to the Image Files. This path MUST have folders
        """
        super(CustomD, self).__init__()
        self.transform = transform
        for _, _, files in os.walk(folderpath, topdown=False):
            self._filenames_images = [os.path.join(folderpath,f) for f in files]

        #print(sorted(self._filenames_targets))
        #print(self._filenames_images)

    def __getitem__(self, index):
        #print(self._filenames_images[index], self._filenames_targets[index])
        _img = Image.open(self._filenames_images[index]).resize((self.img_size, self.img_size))

        for t in self.transform:
            _img = t(_img)
        return _img

    def __len__(self):
        return len(self._filenames_images)

def CustomDataLoader(Dataloader):

    def __call__(self):
        return Dataloader()


if __name__ == "__main__":
    c = CustomD(r"/home/saikat/PycharmProjects/DCGAN/data/custom_face", 64)

