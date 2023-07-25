#!/usr/bin/bash

SRC_PATH=$1
RESULT_PATH=$2

declare -a arr=("hela-mdamb231-mcf7-lclc" "mdamb231-mcf7" "mdamb231-mcf7-lclc" "preo-hela-lclc" "preo-hela-mdamb231-mcf7-lclc" "preo-hela" "hela-breastcancer-lclc" "breastcancer-lclc" "preo-hela-breastcancer-lclc")

for i in "${arr[@]}"
do
    echo -p $SRC_PATH/0.5hrs/$i/sampled/train-test/ -d $RESULT_PATH/0.5hrs/$i/
    python3 main.py -p $SRC_PATH/0.5hrs/$i/sampled/train-test/ -d $RESULT_PATH/0.5hrs/$i/ -m all -i 1 -v cross_val

    echo -p $SRC_PATH/1hrs/$i/sampled/train-test/ -d $RESULT_PATH/1hrs/$i/
    python3 main.py -p $SRC_PATH/1hrs/$i/sampled/train-test/ -d $RESULT_PATH/1hrs/$i/ -m all -i 1 -v cross_val

    echo -p $SRC_PATH/1.5hrs/$i/sampled/train-test/ -d $RESULT_PATH/1.5hrs/$i/
    python3 main.py -p $SRC_PATH/1.5hrs/$i/sampled/train-test/ -d $RESULT_PATH/1.5hrs/$i/ -m all -i 1 -v cross_val

    echo -p $SRC_PATH/2hrs/$i/sampled/train-test/ -d $RESULT_PATH/2hrs/$i/
    python3 main.py -p $SRC_PATH/2hrs/$i/sampled/train-test/ -d $RESULT_PATH/2hrs/$i/ -m all -i 1 -v cross_val

    echo -p $SRC_PATH/2.5hrs/$i/sampled/train-test/ -d $RESULT_PATH/2.5hrs/$i/
    python3 main.py -p $SRC_PATH/2.5hrs/$i/sampled/train-test/ -d $RESULT_PATH/2.5hrs/$i/ -m all -i 1 -v cross_val
done
