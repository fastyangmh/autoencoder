# import
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim

# def


def data_preparation(root='./data', batch_size=1000):
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = MNIST(root=root, train=True, transform=transform, download=True)
    testset = MNIST(root=root, train=False, transform=transform, download=True)
    train_loader = DataLoader(
        dataset=trainset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=testset, batch_size=batch_size)
    return train_loader, test_loader


def create_model(code_size, lr):
    model = AUTOENCODER(code_size=code_size)
    optimizer = optim.Adam(params=model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    return model, optimizer, criterion

# class


class AUTOENCODER(nn.Module):
    def __init__(self, code_size=2):
        super(AUTOENCODER, self).__init__()
        self.enc_l1 = nn.Linear(28*28, 256)
        self.enc_l2 = nn.Linear(256, code_size)
        self.dec_l1 = nn.Linear(code_size, 256)
        self.dec_l2 = nn.Linear(256, 28*28)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.sigmoid(self.enc_l1(x))
        x = self.tanh(self.enc_l2(x))
        x = self.sigmoid(self.dec_l1(x))
        x = self.sigmoid(self.dec_l2(x))
        return x

    def encoder(self, x):
        x = self.sigmoid(self.enc_l1(x))
        x = self.tanh(self.enc_l2(x))
        return x

    def decoder(self, x):
        x = self.sigmoid(self.dec_l1(x))
        x = self.sigmoid(self.dec_l2(x))
        return x


if __name__ == "__main__":
    # parameters
    lr = 0.001
    niter = 100
    code_size = 2

    # get dataloader
    train_loader, test_loader = data_preparation()

    # display first image in train_loader
    img = train_loader.dataset.data[0]
    plt.imshow(img, cmap='gray')
    plt.show()

    # create model, optimizer, criterion
    model, optimizer, criterion = create_model(code_size=code_size, lr=lr)
