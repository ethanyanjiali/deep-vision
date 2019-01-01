# coding: utf-8
import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchsummary import summary
from torchvision import transforms

from data_load import ImageNet2012Dataset, RandomCrop, Rescale, ToTensor, RandomHorizontalFlip, CenterCrop, Normalize
from models.alexnet_v1 import AlexNetV1
from models.alexnet_v2 import AlexNetV2
from models.vgg16 import VGG16
from models.vgg19 import VGG19
from models.inception_v1 import InceptionV1
from models.resnet34 import ResNet34

evaluate_batch_size = 128
epochs = 250
desired_image_shape = torch.empty(3, 224, 224).size()
model_dir = './saved_models/'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
val_transform = transforms.Compose([
    Rescale(256),
    CenterCrop(224),
    ToTensor(),
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L195
    # this is pre-calculated mean and std of imagenet dataset
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


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
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=16)

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
        num_workers=16)

    return val_loader


def calc_accuracy_top1(output, Y):
    max_vals, max_indices = torch.max(output, 1)
    acc = (max_indices == Y).cpu().sum().data.numpy() / max_indices.size()[0]
    return acc


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


# https://github.com/pytorch/examples/blob/master/imagenet/main.py#L381
def accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def evaluate(net, criterion, epoch, val_loader, acc_logger):
    net.eval()
    total_loss = 0
    top1_acc = 0.0
    top5_acc = 0.0
    # turn off grad to avoid cuda out of memory error
    with torch.no_grad():
        for batch_i, data in enumerate(val_loader):
            images = data['image']
            annotations = data['annotation']
            annotations = annotations.to(device=device, dtype=torch.long)
            images = images.to(device=device, dtype=torch.float)

            output = net(images)
            loss = criterion(output, annotations)
            acc1, acc5 = accuracy(output, annotations, topk=(1, 5))
            top1_acc += acc1[0]
            top5_acc += acc5[0]
            total_loss += loss

        top1_acc = top1_acc / len(val_loader)
        top5_acc = top5_acc / len(val_loader)
        acc_logger.append(top1_acc)
        print('Epoch: {}, Validation Top 1 acc: {}'.format(epoch, top1_acc))
        print('Epoch: {}, Validation Top 5 acc: {}'.format(epoch, top5_acc))
        val_loss = total_loss / len(val_loader)
        print('Epoch: {}, Validation Set Loss: {}'.format(epoch, val_loss))
        return val_loss, top1_acc, top5_acc


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

        if isinstance(net, InceptionV1):
            # forward propagation - calculate the output
            output, aux1_output, aux2_output = net(images)

            # calculate the loss
            loss = criterion(output, annotations)
            aux1_loss = criterion(output, annotations)
            aux2_loss = criterion(output, annotations)
            loss = loss + aux1_loss * 0.3 + aux2_loss * 0.3

            optimizer.zero_grad()

            loss.backward()
        else:
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
        lr = get_lr(optimizer)
        optimizer.step()

        # adjust the running loss
        batches_loss += loss.item()

        if batch_i % 10 == 9:  # print every 10 batches
            print(
                'Time, {}, Epoch: {}, Batch: {}, Avg. Loss: {}, LR: {}'.format(
                    time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
                    epoch, batch_i + 1, batches_loss / 10, lr))
            loss_logger.append(batches_loss)
            batches_loss = 0.0


def start(model_name, net, criterion, optimizer, transform, batch_size,
          start_epoch, loss_logger, acc_logger, scheduler):
    print("CUDA is available: {}".format(torch.cuda.is_available()))

    # loader will split datatests into batches witht size defined by batch_size
    train_loader = initialize_train_loader(transform, batch_size)
    val_loader = initialize_validation_loader(val_transform)

    model_id = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())

    net.to(device=device)
    summary(net, (3, 224, 224))

    # make initial evaluation
    val_loss, top1_acc, top5_acc = evaluate(net, criterion, start_epoch - 1,
                                            val_loader, acc_logger)

    for i in range(start_epoch, epochs + 1):
        checkpoint_file = '{}-{}-epoch-{}.pt'.format(model_name, model_id, i)

        # for regular scheduler, we step before trainning and evaludation
        if scheduler is not None and not isinstance(
                scheduler,
                optim.lr_scheduler.ReduceLROnPlateau,
        ):
            scheduler.step()

        # train all data for one epoch
        train(net, criterion, optimizer, i, train_loader, model_id,
              loss_logger)

        # evaludate the accuracy after each epoch
        val_loss, top1_acc, top5_acc = evaluate(net, criterion, i, val_loader,
                                                acc_logger)

        # for regular ReduceLROnPlateau scheduler, we step after trainning and evaludation
        if scheduler is not None and isinstance(
                scheduler,
                optim.lr_scheduler.ReduceLROnPlateau,
        ):
            scheduler.step(val_loss)

        # https://discuss.pytorch.org/t/loading-a-saved-model-for-continue-training/17244/3
        # https://github.com/pytorch/pytorch/issues/2830
        # https://pytorch.org/tutorials/beginner/saving_loading_models.html
        torch.save({
            'epoch':
            i,
            'model':
            net.state_dict(),
            'optimizer':
            optimizer.state_dict(),
            'scheduler':
            scheduler.state_dict() if scheduler is not None else None,
            'loss_logger':
            loss_logger,
            'acc_logger':
            acc_logger,
        }, model_dir + checkpoint_file)

    print("Finished training!")
    checkpoint_file = '{}-{}-final.pt'.format(model_name, model_id)
    torch.save({
        'epoch':
        epochs,
        'model':
        net.state_dict(),
        'optimizer':
        optimizer.state_dict(),
        'scheduler':
        scheduler.state_dict() if scheduler is not None else None,
        'loss_logger':
        loss_logger,
        'acc_logger':
        acc_logger,
    }, model_dir + checkpoint_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        choices=[
            "alexnet1",
            "alexnet2",
            "vgg16",
            "vgg19",
            "inception1",
            "resnet34",
        ],
        help="specify model name",
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        type=str,
        help="specify checkpoint file path",
    )
    args = parser.parse_args()
    model_name = args.model
    checkpoint_file = args.checkpoint
    scheduler = None

    transform = transforms.Compose([
        Rescale(256),
        RandomHorizontalFlip(0.5),
        RandomCrop(224),
        ToTensor(),
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L195
        # this is pre-calculated mean and std of imagenet dataset
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if model_name == "alexnet1":
        batch_size = 128
        # instantiate the neural network
        net = AlexNetV1()
        # define the loss function using CrossEntropyLoss
        criterion = nn.CrossEntropyLoss()
        # define the params updating function using SGD
        optimizer = optim.SGD(
            net.parameters(),
            lr=0.01,
            momentum=0.9,
            weight_decay=0.0005,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            "min",
            factor=0.1,
        )
    elif model_name == "alexnet2":
        # "We trained our models using stochastic gradient descent with a batch size of 128 examples" alexnet1.[1]
        batch_size = 128
        # instantiate the neural network
        net = AlexNetV2()
        # define the loss function using CrossEntropyLoss
        criterion = nn.CrossEntropyLoss()
        # define the params updating function using SGD
        # loss will become nan if init lr = 0.01
        optimizer = optim.SGD(
            net.parameters(),
            lr=0.01,
            momentum=0.9,
            weight_decay=0.0005,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            "min",
            factor=0.1,
        )
    elif model_name == "vgg16":
        # instantiate the neural network
        net = VGG16()
        # define the loss function using CrossEntropyLoss
        criterion = nn.CrossEntropyLoss()
        # "The batch size was set to 256, momentum to 0.9. The training was regularised by
        # weight decay (the L2 penalty multiplier set to 5*10^4) and dropout regularisation
        # for the first two fully-connected layers (dropout ratio set to 0.5).
        # The learning rate was initially set to 10−2" vgg16.[1]

        # "Multi-GPU training exploits data parallelism, and is carried out by splitting each batch of
        # training images into several GPU batches, processed in parallel on each GPU." vgg16[1]

        # However, since I'm training on one GPU, to avoid "CUDA out of memory" issue, I have to reduce the
        # batch size here
        # using lr=0.01 with kaiming init will result in loss explosion
        # 35.06133270263672
        # 115.53974151611328
        # 41985.546875
        # 2.7532622521468985e+29
        # nan
        batch_size = 128
        optimizer = optim.SGD(
            net.parameters(),
            lr=0.01,
            momentum=0.9,
            weight_decay=0.0005,
        )
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=10,
            gamma=0.5,
        )
    elif model_name == "vgg19":
        # instantiate the neural network
        net = VGG19()
        # define the loss function using CrossEntropyLoss
        criterion = nn.CrossEntropyLoss()
        # "The batch size was set to 256, momentum to 0.9. The training was regularised by
        # weight decay (the L2 penalty multiplier set to 5^10−4) and dropout regularisation
        # for the first two fully-connected layers (dropout ratio set to 0.5).
        # The learning rate was initially set to 10−2" vgg19.[1]

        # Similar constraints like VGG16 above
        batch_size = 128
        optimizer = optim.SGD(
            net.parameters(),
            lr=0.01,
            momentum=0.9,
            weight_decay=0.0005,
        )
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=10,
            gamma=0.5,
        )
    elif model_name == "inception1":
        # instantiate the neural network
        net = InceptionV1()
        # define the loss function using CrossEntropyLoss
        criterion = nn.CrossEntropyLoss()
        batch_size = 128
        # "Our training used asynchronous stochastic gradient descent with 0.9 momentum [17],
        # fixed learning rate schedule (decreasing the learning rate by 4% every 8 epochs).
        # Polyak averaging [13] was used to create the final model used at inference time."[1]
        optimizer = optim.SGD(
            net.parameters(),
            lr=0.01,
            momentum=0.9,
            weight_decay=0.0002,
        )
        # However, the original lr schedule requires 250 epochs, and it stays at loss=3.5 because lr is going down too slowly
        # As reported by https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet
        max_epochs = 60.0
        power = 0.5
        # Hence I use poly lr policy as recommended here:
        # https://github.com/BVLC/caffe/blob/master/models/bvlc_googlenet/quick_solver.prototxt#L8
        lr_func = lambda epoch: (1 - epoch / max_epochs)**power if epoch < max_epochs else 0.01
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)
    elif model_name == "resnet34":
        # instantiate the neural network
        net = ResNet34()

        # https://github.com/bearpaw/pytorch-classification/issues/27
        if torch.cuda.device_count() > 1:
            print("Using", torch.cuda.device_count(), "GPUs!")
            net = nn.DataParallel(net)

        # define the loss function using CrossEntropyLoss
        criterion = nn.CrossEntropyLoss()

        # "We use SGD with a mini-batch size of 256."
        # Aslo from Kaiming's disclaimer here https://github.com/KaimingHe/deep-residual-networks#disclaimer-and-known-issues
        # Since he uses 8 GPU and 32 for each GPU
        # So for ResNet I will be using 8 Nvidia K80 GPU instead
        # Note that this batch size won't fit on single P100 16G GPU
        batch_size = 512
        # "The learning rate starts from 0.1 and is divided by 10 when the error plateaus,
        # We use a weight decay of 0.0001 and a momentum of 0.9."" resnet34.[1]
        optimizer = optim.SGD(
            net.parameters(),
            lr=0.1,
            momentum=0.9,
            weight_decay=0.0001,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            "min",
            factor=0.1,
        )

    start_epoch = 1
    loss_logger = []
    acc_logger = []
    if checkpoint_file:
        checkpoint = torch.load(checkpoint_file)
        net.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        # https://github.com/pytorch/pytorch/issues/2830#issuecomment-336194949
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
        start_epoch = checkpoint['epoch'] + 1
        loss_logger = checkpoint['loss_logger']

        # load scheduler state if exist
        if scheduler is not None:
            scheduler_state = checkpoint.get('scheduler')
            if scheduler_state is not None:
                scheduler.load_state_dict(scheduler_state)

        # load accuracy logger if exist. If not, initialize it with 0s
        acc_logger = checkpoint.get('acc_logger')
        if acc_logger is None:
            acc_logger = [0 for i in range(start_epoch)]

    start(model_name, net, criterion, optimizer, transform, batch_size,
          start_epoch, loss_logger, acc_logger, scheduler)
