#!/usr/bin/env python
# coding: utf-8

# In[10]:


import matplotlib.pyplot as plt

log_file1 = "./logs/vgg16-2018-12-09-19-15-09.log"
log_file2 = "./logs/alexnet2-2018-12-11-07-58-29.log"
batches1 = []
losses1 = []
with open(log_file1, 'r') as f:
    line = f.readline()
    cnt = 0
    while line:
        if line.startswith("Time"):
            parts = line.split(",")
            cnt += 1
            batches1.append(cnt)
            losses1.append(float(parts[4][12:-1]))
        line = f.readline()
batches2 = []
losses2 = []
with open(log_file2, 'r') as f:
    line = f.readline()
    cnt = 0
    while line:
        if line.startswith("Time"):
            parts = line.split(",")
            cnt += 1
            batches2.append(cnt)
            losses2.append(float(parts[4][12:-1]))
        line = f.readline()

plt.figure(figsize=(20,5))
plt.subplot(121)
plt.title('vgg16')
plt.plot(batches1, losses1)
plt.subplot(122)
plt.title('alexnet2')
plt.plot(batches2, losses2)
plt.show()


# In[ ]:




