mount-imagenet:
	sudo mount -o discard,defaults /dev/sdb ./dataset

train_alexnet1:
	nohup python -u train.py -m alexnet1 &

train_alexnet2:
	nohup python -u train.py -m alexnet2 &