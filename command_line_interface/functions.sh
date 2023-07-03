#!/bin/sh


score_to_grade() {
    if [ $1 -ge 90 ]
    then
        echo A
    elif [ $1 -ge 80 ] && [ $1 -le 89 ]
    then
        echo B
    elif [ $1 -ge 70 ] && [ $1 -le 79 ]
    then
        echo C
    else
        echo F
    fi
}