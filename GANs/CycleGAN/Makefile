train:
	nohup python -u main.py --dataset=horse2zebra --batch_size=8 &

ps:
	ps -ef | grep python

tb:
	tensorboard --logdir=./logs

gpu:
	nvidia-smi -l 1