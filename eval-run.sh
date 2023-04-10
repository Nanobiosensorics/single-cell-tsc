#!/usr/bin/bash

python3 eval.py -p ../cell_data/data/1d-class-datasets/hela-mdamb231-mcf7-lclc/fibronectin_full/ -d ./results/hela-mdamb231-mcf7-lclc/ -c ../cell_data/data/1d-class-datasets/
python3 eval.py -p ../cell_data/data/1d-class-datasets/mdamb231-mcf7/fibronectin_full/ -d ./results/mdamb231-mcf7/ -c ../cell_data/data/1d-class-datasets/
python3 eval.py -p ../cell_data/data/1d-class-datasets/mdamb231-mcf7-lclc/fibronectin_full/ -d ./results/mdamb231-mcf7-lclc/ -c ../cell_data/data/1d-class-datasets/
python3 eval.py -p ../cell_data/data/1d-class-datasets/preo-hela-lclc/fibronectin_full/ -d ./results/preo-hela-lclc/ -c ../cell_data/data/1d-class-datasets/
python3 eval.py -p ../cell_data/data/1d-class-datasets/preo-hela-mdamb231-mcf7-lclc/fibronectin_full/ -d ./results/preo-hela-mdamb231-mcf7-lclc/ -c ../cell_data/data/1d-class-datasets/
