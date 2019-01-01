from data_load import MnistDataset

def run_epochs():
    for i in range(start_epoch, epochs + 1):

def train():
    train_dataset = MnistDataset(
        images_path='./dataset/train-images-idx3-ubyte',
        labels_path='./dataset/train-labels-idx1-ubyte',
    )
    for batch_i, data in enumerate(train_loader):
         

def validate():

if __name__ == "__main__":
    train()