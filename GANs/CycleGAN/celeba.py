from shutil import copyfile

with open('./list_attr_celeba.txt') as fp:
    line = fp.readline()
    line = fp.readline()
    line = fp.readline()
    while line:
        filename = line[0:10]
        gender = line[70:73]
        print(gender)
        if int(gender) == -1:
            print(filename, 'female')
            copyfile('./img_align_celeba/{}'.format(filename), './datasets/celeba/trainB/{}'.format(filename))
        elif int(gender) == 1:
            print(filename, 'male')
            copyfile('./img_align_celeba/{}'.format(filename), './datasets/celeba/trainA/{}'.format(filename))
        else:
            raise Exception('Unknown gender')
        line = fp.readline()

