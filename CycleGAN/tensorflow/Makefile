train_horse:
	nohup python -u main.py --dataset=horse2zebra --batch_size=4 &

train_celeba:
	nohup python -u main.py --dataset=celeba --batch_size=4 &

train_celeba_p:
	nohup python -u main.py --dataset=celeba_partial --batch_size=4 &

ps:
	ps -ef | grep python

tb:
	tensorboard --logdir=./logs

gpu:
	nvidia-smi -l 1