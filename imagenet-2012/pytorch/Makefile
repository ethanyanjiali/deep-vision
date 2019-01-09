TIME := `/bin/date "+%Y-%m-%d-%H-%M-%S"`

find-python:
	ps -ef | grep python

mount-imagenet:
	mkdir -p ./dataset
	sudo mount --read-only -o norecovery /dev/sdb ../dataset

mount-imagenet-aws:
	mkdir -p ./dataset
	sudo mount -o norecovery /dev/xvdb ../dataset

train_alexnet1:
	mkdir -p ./saved_models
	nohup python -u train.py -m alexnet1 > "alexnet1-${TIME}".log &

train_alexnet2:
	mkdir -p ./saved_models
	nohup python -u train.py -m alexnet2 > "alexnet2-${TIME}".log &

resume_alexnet2:
	mkdir -p ./saved_models
	nohup python -u train.py -m alexnet2 -c /home/yanjia.li/cv-classification/imagenet-2012/pytorch/saved_models/alexnet2-2019-01-01T05:56:51-epoch-80.pt > "alexnet2-${TIME}".log &

train_vgg16:
	mkdir -p ./saved_models
	nohup python -u train.py -m vgg16 > "vgg16-${TIME}".log &

resume_vgg16:
	mkdir -p ./saved_models
	nohup python -u train.py -m vgg16 -c /home/yanjia.li/cv-classification/imagenet-2012/pytorch/saved_models/vgg16-2019-01-01T05:56:32-epoch-27.pt > "vgg16-${TIME}".log &

train_vgg19:
	mkdir -p ./saved_models
	nohup python -u train.py -m vgg19 > "vgg19-${TIME}".log &

train_inception1:
	mkdir -p ./saved_models
	nohup python -u train.py -m inception1 > "inception1-${TIME}".log &

resume_inception1:
	mkdir -p ./saved_models
	nohup python -u train.py -m inception1 -c /home/ubuntu/cv-classification/imagenet-2012/pytorch/saved_models/inception1-2018-12-30T07:26:12-epoch-43.pt > "inception1-${TIME}".log &

train_resnet34:
	mkdir -p ./saved_models
	nohup python -u train.py -m resnet34 > "resnet34-${TIME}".log &

resume_resnet34:
	mkdir -p ./saved_models
	nohup python -u train.py -m resnet34 -c /home/ubuntu/cv-classification/imagenet-2012/pytorch/saved_models/resnet34-2019-01-01T05:47:30-epoch-58.pt > "resnet34-${TIME}".log &

train_resnet50:
	mkdir -p ./saved_models
	nohup python -u train.py -m resnet50 > "resnet50-${TIME}".log &

resume_resnet50:
	mkdir -p ./saved_models
	nohup python -u train.py -m resnet50 -c /home/ubuntu/cv-classification/imagenet-2012/pytorch/saved_models/resnet50-2019-01-08T07:17:18-epoch-97.pt > "resnet50-${TIME}".log &

train_mobilenet1:
	mkdir -p ./saved_models
	nohup python -u train.py -m mobilenet1 > "mobilenet1-${TIME}".log &

visualize:
	jupyter notebook ./notebooks