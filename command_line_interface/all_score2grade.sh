#!/bin/sh

source functions.sh

while read line
do
    name=$(echo $line | cut -d "," -f1)
    score=$(echo $line | cut -d "," -f2)

    grade=$(score_to_grade $score)

    echo $name : $grade
done < input.txt