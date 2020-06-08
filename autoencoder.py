# import
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torch
import numpy as np
import random

# def


def set_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)


def data_preparation(root='./data', batch_size=1000):
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = MNIST(root=root, train=True, transform=transform, download=True)
    testset = MNIST(root=root, train=False, transform=transform, download=True)
    train_loader = DataLoader(
        dataset=trainset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(
        dataset=testset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def create_model(code_size, lr, use_cuda):
    model = AUTOENCODER(code_size=code_size)
    optimizer = optim.Adam(params=model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    if use_cuda:
        model = model.cuda()
    return model, optimizer, criterion


def train(dataloader, model, optimizer, criterion, niter, use_cuda):
    history = []
    for it in tqdm(range(1, niter+1)):
        model.train()
        total_loss = []
        for x, _ in dataloader:
            if use_cuda:
                x = x.cuda()
            x = x.view(-1, 28*28)
            xhat = model(x)
            loss = criterion(xhat, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss.append(loss.item())
        history.append(np.mean(total_loss))
        print('Training loss: {}'.format(round(history[-1], 4)))
    return history


def evaluate(dataloader, model, use_cuda, num_show=10):
    for x, _ in dataloader:
        break
    x = x[:num_show]
    if use_cuda:
        x = x.cuda()
    model.eval()
    with torch.no_grad():
        xhat = model(x.view(-1, 28*28)).cpu().data.numpy()
    plt.figure(figsize=(int(num_show*1.5), int(num_show*0.3)))
    for idx in range(num_show):
        plt.subplot(2, num_show, 1+idx)
        plt.imshow(x[idx].view(28, 28).cpu().data.numpy(), cmap='gray')
        plt.subplot(2, num_show, 1+idx+num_show)
        plt.imshow(xhat[idx].reshape(28, 28), cmap='gray')
    plt.tight_layout()
    plt.show()


# class


class AUTOENCODER(nn.Module):
    def __init__(self, code_size=2):
        super(AUTOENCODER, self).__init__()
        self.enc_l1 = nn.Linear(28*28, code_size)
        self.dec_l1 = nn.Linear(code_size, 28*28)
        self.dec_l2 = nn.Linear(28*28, 28*28)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.tanh(self.enc_l1(x))
        x = self.relu(self.dec_l1(x))
        x = self.relu(self.dec_l2(x))
        return x

    def encoder(self, x):
        x = self.tanh(self.enc_l1(x))
        return x

    def decoder(self, x):
        x = self.relu(self.dec_l1(x))
        x = self.relu(self.dec_l2(x))
        return x


if __name__ == "__main__":
    # parameters
    random_seed = 0
    root = './data'
    batch_size = 32
    code_size = 2
    lr = 2e-3
    use_cuda = torch.cuda.is_available()
    niter = 20
    num_show = 10

    # set seed
    set_seed(random_seed=random_seed)

    # get dataloader
    train_loader, test_loader = data_preparation(
        root=root, batch_size=batch_size)

    # display first image in train_loader
    img = train_loader.dataset.data[0]
    plt.imshow(img, cmap='gray')
    plt.show()

    # create model, optimizer, criterion
    model, optimizer, criterion = create_model(
        code_size=code_size, lr=lr, use_cuda=use_cuda)

    # train model
    history = train(dataloader=train_loader, model=model, optimizer=optimizer,
                    criterion=criterion, niter=niter, use_cuda=use_cuda)

    # evaluate model
    print('Training data')
    evaluate(dataloader=train_loader, model=model,
             use_cuda=use_cuda, num_show=num_show)
    print('\nTest data')
    evaluate(dataloader=test_loader, model=model,
             use_cuda=use_cuda, num_show=num_show)
