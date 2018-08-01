#!/bin/sh

i=1

while [ $i -lt 15 ]
do
    echo $i
    python3.6 train.py --data-dir data --mode 0 --model-type 5 --features $i --treeton-distrib-dir $1
    i=$((i+1))
done