import time

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
from torchsummary import summary

from data_load import (RandomCrop, ImageNet2012Dataset, Rescale, ToTensor)
from models.alexnet1 import AlexNet

train_batch_size = 128
evaluate_batch_size = 32
epochs = 55
desired_image_shape = torch.empty(3, 224, 224).size()
model_dir = './saved_models/'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def initialize_train_loader():
    train_dataset = ImageNet2012Dataset(
        root_dir='../../imagenet-2012/train_flatten/',
        labels_file='../../imagenet-2012/synsets.txt',
    )
    print('Number of train images: ', len(train_dataset))

    assert train_dataset[0]['image'].size(
    ) == desired_image_shape, "Wrong train image dimension"

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=0)

    return train_loader


def initialize_validation_loader():
    val_dataset = ImageNet2012Dataset(
        root_dir='../../imagenet-2012/val_flatten/',
        labels_file='../../imagenet-2012/synsets.txt',
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
            annotations = annotations.to(device=device, dtype=torch.int)
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
            print('Epoch: {}, Batch: {}, Avg. Loss: {}'.format(
                epoch, batch_i + 1, batches_loss / 10))
            loss_logger.append(batches_loss)
            batches_loss = 0.0


def run():
    print("CUDA is available: {}".format(torch.cuda.is_available()))

    # loader will split datatests into batches witht size defined by batch_size
    train_loader = initialize_train_loader()
    val_loader = initialize_validation_loader()

    model_id = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
    # instantiate the neural network
    net = AlexNet()
    net.to(device=device)
    summary(net, (3, 224, 224))
    # define the loss function using CrossEntropyLoss
    criterion = nn.CrossEntropyLoss()
    # define the params updating function using SGD
    optimizer = optim.SGD(
        net.parameters(),
        lr=0.01,
        momentum=0.9,
        weight_decay=0.0005,
    )

    loss_logger = []

    for i in range(1, epochs + 1):
        model_name = 'model-{}-epoch-{}.pt'.format(model_id, i)

        # train all data for one epoch
        train(net, criterion, optimizer, i, train_loader, model_id,
              loss_logger)

        # evaludate the accuracy after each epoch
        evaluate(net, criterion, i, val_loader)

        # save model after every 5 epochs
        # https://discuss.pytorch.org/t/loading-a-saved-model-for-continue-training/17244/3
        # https://github.com/pytorch/pytorch/issues/2830
        # https://pytorch.org/tutorials/beginner/saving_loading_models.html
        if i % 5 == 1:
            torch.save({
                'epoch': i,
                'model': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss_logger': loss_logger,
            }, model_dir + model_name)

    print("Finished training!")
    model_name = 'model-{}-final.pt'.format(model_id)
    torch.save({
        'epoch': epochs,
        'model': net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss_logger': loss_logger,
    }, model_dir + model_name)


if __name__ == "__main__":
    run()
