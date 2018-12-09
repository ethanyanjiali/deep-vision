TIME=$(shell echo $(date +"%Y-%m-%d-%H-%M-%S"))

find-python:
	ps -ef | grep python

mount-imagenet:
	mkdir -p ./dataset
	sudo mount --read-only -o norecovery /dev/sdb ./dataset

train_alexnet1:
	mkdir -p ./saved_models
	nohup python -u train.py -m alexnet1 > "alexnet1-${TIME}".log &

train_alexnet2:
	mkdir -p ./saved_models
	nohup python -u train.py -m alexnet2 > "alexnet2-${TIME}".log &

train_vgg16:
	mkdir -p ./saved_models
	nohup python -u train.py -m vgg16 > "vgg16-${TIME}".log &

train_vgg19:
	mkdir -p ./saved_models
	nohup python -u train.py -m vgg19 > "vgg19-${TIME}".log &