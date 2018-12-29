TIME := `/bin/date "+%Y-%m-%d-%H-%M-%S"`

find-python:
	ps -ef | grep python

mount-imagenet:
	mkdir -p ./dataset
	sudo mount --read-only -o norecovery /dev/sdb ./dataset

mount-imagenet-aws:
	mkdir -p ./dataset
	sudo mount --read-only -o norecovery /dev/xvdb ./dataset

train_alexnet1:
	mkdir -p ./saved_models
	nohup python -u train.py -m alexnet1 > "alexnet1-${TIME}".log &

train_alexnet2:
	mkdir -p ./saved_models
	nohup python -u train.py -m alexnet2 > "alexnet2-${TIME}".log &

resume_alexnet2:
	mkdir -p ./saved_models
	nohup python -u train.py -m alexnet2 -c /home/yanjia.li/image-classification/imagenet-2012/saved_models/alexnet2-2018-12-25T18:04:27-final.pt > "alexnet2-${TIME}".log &

train_vgg16:
	mkdir -p ./saved_models
	nohup python -u train.py -m vgg16 > "vgg16-${TIME}".log &

resume_vgg16:
	mkdir -p ./saved_models
	nohup python -u train.py -m vgg16 -c /home/yanjia.li/image-classification/imagenet-2012/saved_models/vgg16-2018-12-23T22:41:43-epoch-33.pt > "vgg16-${TIME}".log &

train_vgg19:
	mkdir -p ./saved_models
	nohup python -u train.py -m vgg19 > "vgg19-${TIME}".log &

train_inception1:
	mkdir -p ./saved_models
	nohup python -u train.py -m inception1 > "inception1-${TIME}".log &

resume_inception1:
	mkdir -p ./saved_models
	nohup python -u train.py -m inception1 -c /home/ubuntu/image-classification/imagenet-2012/saved_models/inception1-2018-12-24T00:20:14-epoch-30.pt > "inception1-${TIME}".log &

train_resnet34:
	mkdir -p ./saved_models
	nohup python -u train.py -m resnet34 > "resnet34-${TIME}".log &

resume_resnet34:
	mkdir -p ./saved_models
	nohup python -u train.py -m resnet34 -c /home/ubuntu/cv-classification/imagenet-2012/saved_models/resnet34-2018-12-27T05:50:50-epoch-86.pt > "resnet34-${TIME}".log &

visualize:
	jupyter notebook ./notebooks