#!/usr/bin/bash

SRC_PATH=$1
RESULT_PATH=$2

declare -a cell_types=("hela-mdamb231-mcf7-lclc" "mdamb231-mcf7" "mdamb231-mcf7-lclc" "preo-hela-mdamb231-mcf7-lclc" "hela-breastcancer-lclc" "breastcancer-lclc" "preo-hela-breastcancer-lclc")
declare -a times=(30 60 90 120)
for i in "${cell_types[@]}"
do
    for j in "${times[@]}"
    do
        # echo -p $SRC_PATH -d $RESULT_PATH -m all -t $j -tp $i
        python3 main.py -p $SRC_PATH/$i -d $RESULT_PATH -m single -t $j -tp $i -c resnet
    done
done
