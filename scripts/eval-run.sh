#!/usr/bin/bash

SRC_PATH=$1
RESULT_PATH=$2

declare -a cell_types=("hela-lclc-mcf7-mdamb231" "mcf7-mdamb231" "lclc-mcf7-mdamb231" "hela-lclc-mcf7-mdamb231-preo" "breastcancer-hela-lclc" "breastcancer-lclc" "breastcancer-hela-lclc-preo")
declare -a times=(30 60 90 120 150)

for i in "${cell_types[@]}"
do
    for j in "${times[@]}"
    do
        python3 eval.py -p $SRC_PATH/$i/$j/ -d $RESULT_PATH/$i/$j/ -c $SRC_PATH/
    done
done