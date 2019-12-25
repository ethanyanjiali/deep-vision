mkdir ./val_flatten
for a in `ls -1`;
do
        b=$(echo "$a") ;
        for f in `ls -1 ${b}`
        do
            cp "./${b}/${f}" "./val_flatten/${b}_${f}"
        done
done
sudo find ./val_flatten -name 'val_flatten_*' -delete