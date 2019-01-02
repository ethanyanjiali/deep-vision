TIME := `/bin/date "+%Y-%m-%d-%H-%M-%S"`

visualize:
	jupyter notebook ./notebooks

find-python:
	ps -ef | grep python

train_lenet5:
	mkdir -p ./saved_models
	nohup python -u train.py -m lenet5 > "lenet5-${TIME}".log &

resume_lenet5:
	mkdir -p ./saved_models
	nohup python -u train.py -m lenet5 -c ./saved_models/lenet5-2019-01-01T19:48:10-epoch-150.pt > "lenet5-${TIME}".log &