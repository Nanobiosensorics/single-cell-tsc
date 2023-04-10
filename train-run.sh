#!/usr/bin/bash

RESULT_PATH=$1

declare -a arr=("hela-mdamb231-mcf7-lclc" "mdamb231-mcf7" "mdamb231-mcf7-lclc" "preo-hela-lclc" "preo-hela-mdamb231-mcf7-lclc" "preo-hela" "hela-breastcancer-lclc" "breastcancer-lclc" "preo-hela-breastcancer-lclc")

# declare -a arr=("preo-hela" "hela-breastcancer-lclc" "breastcancer-lclc" "preo-hela-breastcancer-lclc")

for i in "${arr[@]}"
do
    echo -p ../cell_data/data/1d-class-datasets/interpolated-2nd-run/1.5hrs/$i/fibronectin_full/ -d $RESULT_PATH/1.5hrs/$i/
    python3 main.py -p ../cell_data/data/1d-class-datasets/interpolated-2nd-run/1.5hrs/$i/fibronectin_full/ -d $RESULT_PATH/1.5hrs/$i/ -m all -i 1
    echo -p ../cell_data/data/1d-class-datasets/interpolated-2nd-run/1.5hrs/$i/fibronectin_full/ -d $RESULT_PATH/1.5hrs/$i/ -c ../cell_data/data/1d-class-datasets/interpolated-2nd-run/1.5hrs/
    python3 eval.py -p ../cell_data/data/1d-class-datasets/interpolated-2nd-run/1.5hrs/$i/fibronectin_full/ -d $RESULT_PATH/1.5hrs/$i/ -c ../cell_data/data/1d-class-datasets/interpolated-2nd-run/1.5hrs/

    echo -p ../cell_data/data/1d-class-datasets/interpolated-2nd-run/2hrs/$i/fibronectin_full/ -d $RESULT_PATH/2hrs/$i/
    python3 main.py -p ../cell_data/data/1d-class-datasets/interpolated-2nd-run/2hrs/$i/fibronectin_full/ -d $RESULT_PATH/2hrs/$i/ -m all -i 1
    echo -p ../cell_data/data/1d-class-datasets/interpolated-2nd-run/2hrs/$i/fibronectin_full/ -d $RESULT_PATH/2hrs/$i/ -c ../cell_data/data/1d-class-datasets/interpolated-2nd-run/2hrs/
    python3 eval.py -p ../cell_data/data/1d-class-datasets/interpolated-2nd-run/2hrs/$i/fibronectin_full/ -d $RESULT_PATH/2hrs/$i/ -c ../cell_data/data/1d-class-datasets/interpolated-2nd-run/2hrs/

    echo -p ../cell_data/data/1d-class-datasets/interpolated-2nd-run/2.5hrs/$i/fibronectin_full/ -d $RESULT_PATH/2.5hrs/$i/
    python3 main.py -p ../cell_data/data/1d-class-datasets/interpolated-2nd-run/2.5hrs/$i/fibronectin_full/ -d $RESULT_PATH/2.5hrs/$i/ -m all -i 1
    echo -p ../cell_data/data/1d-class-datasets/interpolated-2nd-run/2.5hrs/$i/fibronectin_full/ -d $RESULT_PATH/2.5hrs/$i/ -c ../cell_data/data/1d-class-datasets/interpolated-2nd-run/2.5hrs/
    python3 eval.py -p ../cell_data/data/1d-class-datasets/interpolated-2nd-run/2.5hrs/$i/fibronectin_full/ -d $RESULT_PATH/2.5hrs/$i/ -c ../cell_data/data/1d-class-datasets/interpolated-2nd-run/2.5hrs/
done
