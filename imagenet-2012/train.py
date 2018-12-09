import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchsummary import summary
from torchvision import transforms

from data_load import ImageNet2012Dataset, RandomCrop, Rescale, ToTensor
from models.alexnet1 import AlexNet1
from models.alexnet2 import AlexNet2

evaluate_batch_size = 32
epochs = 55
desired_image_shape = torch.empty(3, 224, 224).size()
model_dir = './saved_models/'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def initialize_train_loader(transform, batch_size):
    train_dataset = ImageNet2012Dataset(
        root_dir='./dataset/train_flatten/',
        labels_file='./dataset/synsets.txt',
        transform=transform,
    )
    print('Number of train images: ', len(train_dataset))

    assert train_dataset[0]['image'].size(
    ) == desired_image_shape, "Wrong train image dimension"

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    return train_loader


def initialize_validation_loader(transform):
    val_dataset = ImageNet2012Dataset(
        root_dir='./dataset/val_flatten/',
        labels_file='./dataset/synsets.txt',
        transform=transform,
    )
    print('Number of validation images: ', len(val_dataset))

    assert val_dataset[0]['image'].size(
    ) == desired_image_shape, "Wrong validation image dimension"

    val_loader = DataLoader(
        val_dataset,
        batch_size=evaluate_batch_size,
        shuffle=True,
        num_workers=0)

    return val_loader


def evaluate(net, criterion, epoch, val_loader):
    net.eval()
    total_loss = 0
    # turn off grad to avoid cuda out of memory error
    with torch.no_grad():
        for batch_i, data in enumerate(val_loader):
            images = data['image']
            annotations = data['annotation']
            annotations = annotations.to(device=device, dtype=torch.long)
            images = images.to(device=device, dtype=torch.float)

            output = net(images)
            loss = criterion(output, annotations)

            total_loss += loss
        print('Epoch: {}, Test Dataset Loss: {}'.format(
            epoch, total_loss / len(val_loader)))


def train(net, criterion, optimizer, epoch, train_loader, model_id,
          loss_logger):
    # mark as train mode
    net.train()
    # initialize the batch_loss to help us understand
    # the performance of multiple batches
    batches_loss = 0.0
    print("Start training epoch {}".format(epoch))
    for batch_i, data in enumerate(train_loader):
        # a batch of images (batch_size, 3, 224, 224)
        images = data['image']
        # a batch of keypoints (batch_size, 1)
        annotations = data['annotation']
        # annotation is an integer index
        annotations = annotations.to(device=device, dtype=torch.long)
        # PyTorch likes float type for image. So we convert to it.
        images = images.to(device=device, dtype=torch.float)

        # forward propagation - calculate the output
        output = net(images)
        # calculate the loss
        loss = criterion(output, annotations)

        # https://discuss.pytorch.org/t/why-do-we-need-to-set-the-gradients-manually-to-zero-in-pytorch/4903/8
        # https://stackoverflow.com/questions/44732217/why-do-we-need-to-explicitly-call-zero-grad
        # zero the parameter (weight) gradients
        optimizer.zero_grad()

        loss.backward()
        # https://discuss.pytorch.org/t/how-are-optimizer-step-and-loss-backward-related/7350
        optimizer.step()

        # adjust the running loss
        batches_loss += loss.item()

        if batch_i % 10 == 9:  # print every 10 batches
            print(
                'Time, {}, Epoch: {}, Batch: {}, Avg. Loss: {}'.format(
                    time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
                    epoch, batch_i + 1, batches_loss / 10), )
            loss_logger.append(batches_loss)
            batches_loss = 0.0


def start(model_name, net, criterion, optimizer, transform, batch_size):
    print("CUDA is available: {}".format(torch.cuda.is_available()))

    # loader will split datatests into batches witht size defined by batch_size
    train_loader = initialize_train_loader(transform, batch_size)
    val_loader = initialize_validation_loader(transform)

    model_id = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
    net.to(device=device)
    summary(net, (3, 224, 224))
    loss_logger = []

    for i in range(1, epochs + 1):
        checkpoint_file = '{}-{}-epoch-{}.pt'.format(model_name, model_id, i)

        # train all data for one epoch
        train(net, criterion, optimizer, i, train_loader, model_id,
              loss_logger)

        # evaludate the accuracy after each epoch
        evaluate(net, criterion, i, val_loader)

        # save model after every 2 epochs
        # https://discuss.pytorch.org/t/loading-a-saved-model-for-continue-training/17244/3
        # https://github.com/pytorch/pytorch/issues/2830
        # https://pytorch.org/tutorials/beginner/saving_loading_models.html
        if i % 2 == 1:
            torch.save({
                'epoch': i,
                'model': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss_logger': loss_logger,
            }, model_dir + checkpoint_file)

    print("Finished training!")
    checkpoint_file = '{}-{}-final.pt'.format(model_name, model_id)
    torch.save({
        'epoch': epochs,
        'model': net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss_logger': loss_logger,
    }, model_dir + checkpoint_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        choices=["alexnet1", "alexnet2"],
        help="specify model name",
    )
    args = parser.parse_args()
    model_name = args.model

    if model_name == "alexnet1":
        transform = transforms.Compose([
            Rescale(255),
            RandomCrop(224),
            ToTensor(),
        ])
        batch_size = 128
        # instantiate the neural network
        net = AlexNet1()
        # define the loss function using CrossEntropyLoss
        criterion = nn.CrossEntropyLoss()
        # define the params updating function using SGD
        optimizer = optim.SGD(
            net.parameters(),
            lr=0.01,
            momentum=0.9,
            weight_decay=0.0005,
        )
    elif model_name == "alexnet2":
        transform = transforms.Compose([
            Rescale(255),
            RandomCrop(224),
            ToTensor(),
        ])
        # "We trained our models using stochastic gradient descent with a batch size of 128 examples" alexnet1.[1]
        batch_size = 128
        # instantiate the neural network
        net = AlexNet2()
        # define the loss function using CrossEntropyLoss
        criterion = nn.CrossEntropyLoss()
        # define the params updating function using SGD
        optimizer = optim.SGD(
            net.parameters(),
            lr=0.01,
            momentum=0.9,
            weight_decay=0.0005,
        )

    start(model_name, net, criterion, optimizer, transform, batch_size)
