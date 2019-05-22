TIME := `/bin/date "+%Y-%m-%d-%H-%M-%S"`

find-python:
	ps -ef | grep python

visualize:
	jupyter notebook ./notebooks

train_lenet5:
	mkdir -p ./saved_models
	nohup python -u train.py -m lenet5 > "lenet5-${TIME}".log &

tensorboard:
	tensorboard --logdir=./tensorboard_logs