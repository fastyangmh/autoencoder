# import
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# def


def data_preparation(root='./data', batch_size=1000):
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = MNIST(root=root, train=True, transform=transform, download=True)
    testset = MNIST(root=root, train=False, transform=transform, download=True)
    train_loader = DataLoader(
        dataset=trainset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=testset, batch_size=batch_size)
    return train_loader, test_loader


if __name__ == "__main__":
    # get dataloader
    train_loader, test_loader = data_preparation()

    # display first image in train_loader
    img = train_loader.dataset.data[0]
    plt.imshow(img, cmap='gray')
    plt.show()

    