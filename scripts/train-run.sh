#!/usr/bin/bash

SRC_PATH=$1
RESULT_PATH=$2

declare -a cell_types=("hela-lclc-mcf7-mdamb231" "hela-lclc-mcf7-mdamb231-preo" "breastcancer-hela-lclc"  "breastcancer-hela-lclc-preo" "breastcancer-hela" "hela-mcf7-mdamb231" "mcf7-mdamb231")
declare -a times=(30 60 90 120 150)
for i in "${cell_types[@]}"
do
    for j in "${times[@]}"
    do
        # echo -p $SRC_PATH -d $RESULT_PATH -m all -t $j -tp $i
        python3 main.py -p $SRC_PATH/$i -d $RESULT_PATH -m all -t $j -tp $i
    done
done
