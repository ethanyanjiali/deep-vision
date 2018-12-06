for a in `ls -1 *.tar`;
do
        b=$(echo "$a" | sed 's/\.[^.]*$//') ;
        tar xvf "$a" -C ./"$b"
done