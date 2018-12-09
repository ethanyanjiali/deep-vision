mount-imagenet:
	mkdir -p ./dataset
	sudo mount --read-only -o norecovery /dev/sdb ./dataset

train_alexnet1:
	rm -rf nohup.out
	mkdir -p ./saved_models
	nohup python -u train.py -m alexnet1 &

train_alexnet2:
	rm -rf nohup.out
	mkdir -p ./saved_models
	nohup python -u train.py -m alexnet2 &