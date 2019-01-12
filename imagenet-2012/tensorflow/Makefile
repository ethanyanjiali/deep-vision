TIME := `/bin/date "+%Y-%m-%d-%H-%M-%S"`

find-python:
	ps -ef | grep python

mount-imagenet:
	mkdir -p ./dataset
	sudo mount --read-only -o norecovery /dev/sdb ../dataset

mount-imagenet-aws:
	mkdir -p ./dataset
	sudo mount -o norecovery /dev/xvdb ../dataset

train_alexnet2:
	mkdir -p ./saved_models
	nohup python -u train.py -m alexnet2 > "alexnet2-${TIME}".log &