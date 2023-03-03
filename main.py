from utils.utils import generate_results_csv
from utils.utils import create_directory
from utils.utils import read_dataset
from utils.utils import transform_mts_to_ucr_format
from utils.utils import visualize_filter
from utils.utils import viz_for_survey_paper
from utils.utils import viz_cam
import argparse
import os
import numpy as np
import pandas as pd
import sys
import sklearn
import utils
from utils.constants import CLASSIFIERS
from utils.constants import ARCHIVE_NAMES
from utils.constants import ITERATIONS
from utils.utils import read_all_datasets
from utils.utils import plot_conf_matrix


def fit_classifier(datasets_dict, dataset_name, classifier_name, output_directory):
    x_train = datasets_dict[dataset_name][0]
    y_train = datasets_dict[dataset_name][1]
    x_test = datasets_dict[dataset_name][2]
    y_test = datasets_dict[dataset_name][3]
    labels = {}
    
    if len(datasets_dict[dataset_name]) > 4:
        labels = list(datasets_dict[dataset_name][4].keys())
    print(labels)
    nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))

    # transform the labels from integers to one hot vectors
    enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
    enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
    y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
    y_test = enc.transform(y_test.reshape(-1, 1)).toarray()

    # save orignal y because later we will use binary
    y_true = np.argmax(y_test, axis=1)

    if len(x_train.shape) == 2:  # if univariate
        # add a dimension to make it multivariate with one dimension 
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    input_shape = x_train.shape[1:]
    classifier = create_classifier(classifier_name, input_shape, nb_classes, output_directory)

    classifier.fit(x_train, y_train, x_test, y_test, y_true)
    
    y_pred = classifier.predict(x_test, y_true, x_train, y_train, y_test, return_df_metrics = False)
    y_pred = np.argmax(y_pred, axis=1)
    y_true_labels = [ labels[i] for i in y_true ]
    y_pred_labels = [ labels[i] for i in y_pred ]
    
    true_pred_values = pd.DataFrame({"true": y_true, "pred": y_pred})
    true_pred_values.to_csv(output_directory + "true-pred-values.csv", index=False)
    
    plot_conf_matrix(y_true_labels, y_pred_labels, labels, output_directory + 'conf_matrix.png') 


def create_classifier(classifier_name, input_shape, nb_classes, output_directory, verbose=True):
    if classifier_name == 'fcn':
        from classifiers import fcn
        return fcn.Classifier_FCN(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'mlp':
        from classifiers import mlp
        return mlp.Classifier_MLP(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'resnet':
        from classifiers import resnet
        return resnet.Classifier_RESNET(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'mcnn':
        from classifiers import mcnn
        return mcnn.Classifier_MCNN(output_directory, verbose)
    if classifier_name == 'tlenet':
        from classifiers import tlenet
        return tlenet.Classifier_TLENET(output_directory, verbose)
    if classifier_name == 'twiesn':
        from classifiers import twiesn
        return twiesn.Classifier_TWIESN(output_directory, verbose)
    if classifier_name == 'encoder':
        from classifiers import encoder
        return encoder.Classifier_ENCODER(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'mcdcnn':
        from classifiers import mcdcnn
        return mcdcnn.Classifier_MCDCNN(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'cnn':  # Time-CNN
        from classifiers import cnn
        return cnn.Classifier_CNN(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'inception':
        from classifiers import inception
        return inception.Classifier_INCEPTION(output_directory, input_shape, nb_classes, verbose)


############################################### main

# change this directory for your machine
# root_dir = '.'

def run(args):
    data_path = args.src_path
    dest_path = args.dst_path
    if args.mode == 'all':
        dataset_name = data_path.strip().split('/')[-2]
        iter_cnt = 3 if args.iter_cnt is None else args.iter_cnt
        print('iters ', iter_cnt)
        for classifier_name in ['fcn', 'mlp', 'resnet', 'encoder', 'mcdcnn', 'cnn', 'inception', 'mcnn']:
            print('classifier_name', classifier_name)

            datasets_dict = read_all_datasets(data_path)
            for i in range(iter_cnt):
                
                output_directory = os.path.join(dest_path, classifier_name, dataset_name + f'_itr_{i}', '')

                print('dataset_name: ', dataset_name, output_directory)

                create_directory(output_directory)

                fit_classifier(datasets_dict, dataset_name, classifier_name, output_directory)

                print('DONE')
                
                # the creation of this directory means
                create_directory(output_directory + '/DONE')

    # # elif sys.argv[1] == 'transform_mts_to_ucr_format':
    # #     transform_mts_to_ucr_format()
    # # elif sys.argv[1] == 'visualize_filter':
    # #     visualize_filter(root_dir)
    # # elif sys.argv[1] == 'viz_for_survey_paper':
    # #     viz_for_survey_paper(root_dir)
    # # elif sys.argv[1] == 'viz_cam':
    # #     viz_cam(root_dir)
    # # elif sys.argv[1] == 'generate_results_csv':
    # #     res = generate_results_csv('results.csv', root_dir)
    # #     print(res.to_string())
    elif args.mode == 'single':
        # this is the code used to launch an experiment on a dataset
        dataset_name = data_path.strip().split('/')[-2]
        classifier_name = args.classifier
            
        output_directory = os.path.join(dest_path, classifier_name, dataset_name, '')
        
        create_directory(output_directory)

        test_dir_df_metrics = output_directory + 'df_metrics.csv'

        print('Method: ', classifier_name)
        
        print(output_directory)
        
        if os.path.exists(test_dir_df_metrics):
            print('Already done')
        else:

            create_directory(output_directory)
            datasets_dict = read_dataset(data_path)

            fit_classifier(datasets_dict, dataset_name, classifier_name, output_directory)

            print('DONE')

            # the creation of this directory means
            create_directory(output_directory + '/DONE')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DL-4-TS')
    parser.add_argument('-p', '--src_path', type=str, help="path to the data source folder", required=True)
    parser.add_argument('-d', '--dst_path', type=str, help="path to the result folder", required=True)
    parser.add_argument('-c', '--classifier', type=str, required=False)
    parser.add_argument('-m', '--mode', type=str, 
                        choices=['single', 'all', 'transform_mts_to_ucr_format', 'visualize_filter', 
                                 'viz_for_survey_paper', 'viz_cam', 'generate_results'], required=True)
    parser.add_argument('-i', '--iter_cnt', type=int, required=False)
    args = parser.parse_args()
    run(args)
