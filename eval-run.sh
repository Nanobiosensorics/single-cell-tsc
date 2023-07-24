#!/usr/bin/bash

SRC_PATH=$1
RESULT_PATH=$2

declare -a arr=( "breastcancer-hela" "mdamb231-mcf7-hela" "preo-hela-mdamb231-mcf7-lclc" "preo-hela-breastcancer-lclc" "hela-mdamb231-mcf7-lclc" "hela-breastcancer-lclc" "mdamb231-mcf7" "mdamb231-mcf7-lclc" )

for i in "${arr[@]}"
do
    echo -p $SRC_PATH/0.5hrs/$i/sampled/train-test/ -d $RESULT_PATH/0.5hrs/$i/ -c $SRC_PATH/0.5hrs/
    python3 eval.py -p $SRC_PATH/0.5hrs/$i/sampled/train-test/ -d $RESULT_PATH/0.5hrs/$i/ -c $SRC_PATH/0.5hrs/

    echo -p $SRC_PATH/1hrs/$i/sampled/train-test/ -d $RESULT_PATH/1hrs/$i/ -c $SRC_PATH/1hrs/
    python3 eval.py -p $SRC_PATH/1hrs/$i/sampled/train-test/ -d $RESULT_PATH/1hrs/$i/ -c $SRC_PATH/1hrs/

    echo -p $SRC_PATH/1.5hrs/$i/sampled/train-test/ -d $RESULT_PATH/1.5hrs/$i/ -c $SRC_PATH/1.5hrs/
    python3 eval.py -p $SRC_PATH/1.5hrs/$i/sampled/train-test/ -d $RESULT_PATH/1.5hrs/$i/ -c $SRC_PATH/1.5hrs/

    echo -p $SRC_PATH/2hrs/$i/sampled/train-test/ -d $RESULT_PATH/2hrs/$i/ -c $SRC_PATH/2hrs/
    python3 eval.py -p $SRC_PATH/2hrs/$i/sampled/train-test/ -d $RESULT_PATH/2hrs/$i/ -c $SRC_PATH/2hrs/

    echo -p $SRC_PATH/2.5hrs/$i/sampled/train-test/ -d $RESULT_PATH/2.5hrs/$i/ -c $SRC_PATH/2.5hrs/
    python3 eval.py -p $SRC_PATH/2.5hrs/$i/sampled/train-test/ -d $RESULT_PATH/2.5hrs/$i/ -c $SRC_PATH/2.5hrs/
done