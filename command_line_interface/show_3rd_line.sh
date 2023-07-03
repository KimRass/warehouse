#!/bin/sh

for i in $(find . -name "*.txt")
do
    n_lines=$(wc -l < $i)
    # echo $n_lines
    if [ $n_lines -ge 3 ]
    then
        number=1
        while read line
        do
            if [ $number = 3 ]
            then
                echo $(basename $i) : $line
            fi
            number=$(($number+1))
        done < $i
    fi
done
