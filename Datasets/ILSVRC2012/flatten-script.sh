mkdir -p ./train_flatten
find ./train -mindepth 2 -type f -exec cp -t ./train_flatten -i '{}' +