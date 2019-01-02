import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchsummary import summary

from data_load import MnistDataset
from models.lenet5 import LeNet5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_dir = './saved_models/'

training_config = {
    'lenet5': {
        'name': 'lenet5',
        'batch_size': 64,
        'num_workers': 2,
        'model': LeNet5,
        'optimizer': optim.SGD,
        'optimizer_params': {
            'lr': 0.1,
            'momentum': 0.9,
            'weight_decay': 0.0005,
        },
        'scheduler': optim.lr_scheduler.ReduceLROnPlateau,
        'scheduler_params': {
            'factor': 0.1,
            'mode': 'max',
        },
        'total_epochs': 200,
    }
}


def initialize_loggers():
    loggers = {
        'train_loss': {
            'epochs': [],
            'value': [],
        },
        'val_loss': {
            'epochs': [],
            'value': [],
        },
        'val_top1_acc': {
            'epochs': [],
            'value': [],
        },
        'val_top5_acc': {
            'epochs': [],
            'value': [],
        }
    }
    return loggers


def log_metrics(loggers, name, value, epoch):
    logger = loggers.get(name)
    logger.get('epochs').append(epoch)
    logger.get('value').append(value)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def load_checkpoint(checkpoint_path, net, optimizer, scheduler, loggers):
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    # https://github.com/pytorch/pytorch/issues/2830#issuecomment-336194949
    if torch.cuda.is_available():
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
    scheduler.load_state_dict(checkpoint['scheduler'])
    start_epoch = checkpoint['epoch'] + 1
    loggers = checkpoint['loggers']

    return net, optimizer, scheduler, loggers, start_epoch


def run_epochs(config, checkpoint_path):
    train_dataset = MnistDataset(
        images_path='./dataset/train-images-idx3-ubyte',
        labels_path='./dataset/train-labels-idx1-ubyte',
        # Global mean and standard deviation of the MNIST dataset
        mean=[0.1307],
        std=[0.3081],
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('batch_size'),
        shuffle=True,
        num_workers=config.get('num_workers'),
    )
    val_dataset = MnistDataset(
        images_path='./dataset/t10k-images-idx3-ubyte',
        labels_path='./dataset/t10k-labels-idx1-ubyte',
        mean=[0.1307],
        std=[0.3081],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get('batch_size'),
        shuffle=False,
        num_workers=config.get('num_workers'),
    )

    # Define the neural network.
    Model = config.get('model')
    net = Model()

    # Print the network structure given 1x32x32 input
    summary(net, (1, 32, 32))

    # Define the loss function. CrossEntrophyLoss is the most common one for classification task.
    criterion = nn.CrossEntropyLoss()

    # Define the optimizer
    Optim = config.get('optimizer')
    optimizer = Optim(
        net.parameters(),
        **config.get('optimizer_params'),
    )

    # Define the scheduler
    Sched = config.get('scheduler')
    scheduler = Sched(
        optimizer,
        **config.get('scheduler_params'),
    )

    loggers = initialize_loggers()

    model_id = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
    model_name = config.get('name')

    start_epoch = 1

    if checkpoint_path is not None:
        net, optimizer, scheduler, loggers, start_epoch = load_checkpoint(
            checkpoint_path,
            net,
            optimizer,
            scheduler,
            loggers,
        )

    validate(val_loader, net, criterion, 0, loggers)

    for epoch in range(start_epoch, config.get('total_epochs') + 1):

        train(
            train_loader,
            net,
            criterion,
            optimizer,
            epoch,
            loggers,
        )

        val_loss, top1_acc, top5_acc = validate(
            val_loader,
            net,
            criterion,
            epoch,
            loggers,
        )

        # for ReduceLROnPlateau scheduler, we need to use val_loss as metric
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(top1_acc)
        else:
            scheduler.step()

        checkpoint_file = '{}-{}-epoch-{}.pt'.format(
            model_name,
            model_id,
            epoch,
        )
        torch.save({
            'epoch': epoch,
            'model': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'loggers': loggers,
        }, model_dir + checkpoint_file)


def train(train_loader, net, criterion, optimizer, epoch, loggers):
    # mark as train mode
    net.train()
    # initialize the batch_loss to help us understand the performance of multiple batches
    batches_loss = 0.0
    print("Start training epoch {}".format(epoch))

    for batch_i, data in enumerate(train_loader):
        # extract images and labels
        image = data.get('image')
        label = data.get('label')

        # annotation is an integer index
        label = label.to(device=device, dtype=torch.long)
        # PyTorch likes float type for image. So we convert to it.
        image = image.to(device=device, dtype=torch.float)

        # forward propagation - calculate the output
        output = net(image)

        # calculate the loss
        loss = criterion(output, label)

        # https://discuss.pytorch.org/t/why-do-we-need-to-set-the-gradients-manually-to-zero-in-pytorch/4903/8
        # https://stackoverflow.com/questions/44732217/why-do-we-need-to-explicitly-call-zero-grad
        # zero the parameter (weight) gradients
        optimizer.zero_grad()

        # back propogate and calculate differentiation
        loss.backward()

        # get current learning rate
        lr = get_lr(optimizer)

        # https://discuss.pytorch.org/t/how-are-optimizer-step-and-loss-backward-related/7350
        # update weights by stepping optimizer
        optimizer.step()

        # accumulate the running loss
        batches_loss += loss.item()

        if batch_i % 10 == 9:  # print every 10 batches
            batches_loss = batches_loss / 10.0
            print('Time, {}, Epoch: {}, Batch: {}, Training Loss: {}, LR: {}'.
                  format(
                      time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
                      epoch,
                      batch_i + 1,  # batch_i start from 0
                      batches_loss,
                      lr,
                  ))

            log_metrics(loggers, 'train_loss', batches_loss, epoch)

            batches_loss = 0.0


def validate(val_loader, net, criterion, epoch, loggers):
    net.eval()
    total_loss = 0
    top1_acc = 0.0
    top5_acc = 0.0
    # turn off grad to avoid cuda out of memory error
    with torch.no_grad():
        for batch_i, data in enumerate(val_loader):
            image = data.get('image')
            label = data.get('label')

            label = label.to(device=device, dtype=torch.long)
            image = image.to(device=device, dtype=torch.float)

            output = net(image)
            loss = criterion(output, label)
            acc1, acc5 = accuracy(output, label, topk=(1, 5))
            top1_acc += acc1[0]
            top5_acc += acc5[0]
            total_loss += loss

    top1_acc = top1_acc / len(val_loader)
    top5_acc = top5_acc / len(val_loader)
    print('Epoch: {}, Validation Top 1 acc: {}'.format(epoch, top1_acc))
    print('Epoch: {}, Validation Top 5 acc: {}'.format(epoch, top5_acc))
    val_loss = total_loss / len(val_loader)
    print('Epoch: {}, Validation Set Loss: {}'.format(epoch, val_loss))

    log_metrics(loggers, 'val_top1_acc', top1_acc, epoch)
    log_metrics(loggers, 'val_top5_acc', top5_acc, epoch)
    log_metrics(loggers, 'val_loss', val_loss, epoch)

    return val_loss, top1_acc, top5_acc


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        choices=[
            "lenet5",
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
    checkpoint_path = args.checkpoint
    config = training_config.get(model_name)
    run_epochs(config, checkpoint_path)