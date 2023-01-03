import torch
import torchvision
from torchvision import transforms

def main():
    train_set = torchvision.datasets.CIFAR10(root='~/datasets/', train=True,
                                                    download=True, transform=transforms.ToTensor())

    data = torch.utils.data.DataLoader(
                train_set, batch_size=16,
                shuffle=False,
                num_workers=4)



    for ind, _ in  enumerate(data):


        # Get training data
        # inputs, labels = data

        inputs, labels = data.dataset[0]
        inputs = inputs.unsqueeze(0)
        labels = labels.unsqueeze(0)

        # Train the network
        # [...]


if __name__ == '__main__':
    main()