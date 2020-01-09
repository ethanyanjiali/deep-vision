```bash
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar -xvf VOCtrainval_11-May-2012.tar
```

In order to download test data, you'll need to login to the test server website first, and download the cookies.txt file from the browser.
One way to download cookies files is to use a Chrome extension called "cookies.txt"
```
wget --load-cookies cookies.txt http://host.robots.ox.ac.uk:8080/eval/downloads/VOC2012test.tar
tar -xvf VOC2012test.tar
```