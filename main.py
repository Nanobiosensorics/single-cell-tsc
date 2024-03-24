from utils.utils import create_directory
from data.load import load_dataset, load_dataset_kfold
import argparse
import os
import numpy as np
import pandas as pd
import sklearn
from utils.utils import plot_conf_matrix
        
def fit_classifier_kfold(data_loader, classifier_name, output_directory, apply_gradient=False):
    
    for n, (X_train, y_train, X_test, y_test, (labels, classes), scaler) in enumerate(data_loader):
        target_dir = os.path.join(output_directory, f'cv_{n}') + '/'
        create_directory(target_dir)
        X_test_org = X_test.copy().reshape((X_test.shape[0], X_test.shape[1], 1))
    
        print(labels, classes)
        nb_classes = len(classes)
        
        y_true = y_test.squeeze().copy()

        # # transform the labels from integers to one hot vectors
        enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
        enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
        y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
        y_test = enc.transform(y_test.reshape(-1, 1)).toarray()

        # save orignal y because later we will use binary
        # y_true = np.argmax(y_test, axis=1)
        
        if apply_gradient:
            X_train = np.gradient(X_train, axis=0)
            X_test = np.gradient(X_test, axis=0)

        if len(X_train.shape) == 2:  # if univariate
            # add a dimension to make it multivariate with one dimension 
            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

        input_shape = X_train.shape[1:]
        classifier = create_classifier(classifier_name, input_shape, nb_classes, target_dir)

        classifier.fit(X_train, y_train, X_test, y_test, y_true)
        
        y_pred = classifier.predict(X_test, y_true, X_train, y_train, y_test, return_df_metrics = False)
        y_pred = np.argmax(y_pred, axis=1)
        output = np.hstack((y_true.reshape(-1, 1), y_pred.reshape(-1, 1), scaler.inverse_transform(X_test_org.squeeze())))
        true_pred_values = pd.DataFrame(output)
        true_pred_values.to_csv(os.path.join(target_dir, 'test_output.csv'), header=False, index=False)

def fit_classifier(dataset, classifier_name, output_directory, apply_gradient=False):
    X_train, y_train, X_test, y_test, (labels, classes), scaler = dataset
    X_test_org = X_test.copy().reshape((X_test.shape[0], X_test.shape[1], 1))

    print(labels, classes)
    nb_classes = len(classes)
    
    y_true = y_test.squeeze().copy()

    # # transform the labels from integers to one hot vectors
    enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
    enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
    y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
    y_test = enc.transform(y_test.reshape(-1, 1)).toarray()

    # save orignal y because later we will use binary
    # y_true = np.argmax(y_test, axis=1)
    if apply_gradient:
        X_train = np.gradient(X_train, axis=0)
        X_test = np.gradient(X_test, axis=0)

    if len(X_train.shape) == 2:  # if univariate
        # add a dimension to make it multivariate with one dimension 
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    input_shape = X_train.shape[1:]
    classifier = create_classifier(classifier_name, input_shape, nb_classes, output_directory)

    classifier.fit(X_train, y_train, X_test, y_test, y_true)
    
    y_pred = classifier.predict(X_test, y_true, X_train, y_train, y_test, return_df_metrics = False)
    y_pred = np.argmax(y_pred, axis=1)
    output = np.hstack((y_true.reshape(-1, 1), y_pred.reshape(-1, 1), scaler.inverse_transform(X_test_org.squeeze())))
    true_pred_values = pd.DataFrame(output)
    true_pred_values.to_csv(os.path.join(output_directory, 'test_output.csv'), header=False, index=False)

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


def run(args):
    data_path = args.src_path
    dest_path = args.dst_path
    if args.mode == 'all':
        classifiers = ['cnn', 'fcn', 'mlp', 'resnet', 'mcdcnn', 'inception']
    else:
        classifiers = [args.classifier]
        
    if args.validation == 'normal':
        dataset = load_dataset(os.path.join(data_path, str(args.time)), resample=True, scale=True)
    else:
        dataset = list(load_dataset_kfold(os.path.join(data_path, str(args.time)), resample=True, scale=True))
    for classifier_name in classifiers:
        print('classifier_name', classifier_name)
        output_directory = os.path.join(dest_path, '-'.join([*args.cell_types]), str(args.time), classifier_name) + '/'

        print(output_directory)

        create_directory(output_directory)
        
        if args.validation == 'normal':        
            test_dir_df_metrics = os.path.join(output_directory, 'df_metrics.csv')
            if os.path.exists(test_dir_df_metrics):
                print('Already done')
            else:
                fit_classifier(dataset, classifier_name, output_directory, apply_gradient=args.gradient)
        else:
            test_dir_df_metrics = os.path.join(output_directory, 'cv0', 'df_metrics.csv')
            if os.path.exists(test_dir_df_metrics):
                print('Already done')
            else:
                fit_classifier_kfold(dataset, classifier_name, output_directory, apply_gradient=args.gradient)

        print('DONE')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DL-4-TS')
    parser.add_argument('-p', '--src_path', type=str, help="path to the data source folder", required=True)
    parser.add_argument('-d', '--dst_path', type=str, help="path to the result folder", required=True)
    parser.add_argument('-c', '--classifier', type=str, required=False)
    parser.add_argument('-v', '--validation', type=str, choices=['normal', 'cross_val'], default='normal', required=False)
    parser.add_argument('-m', '--mode', type=str, 
                        choices=['single', 'all', 'transform_mts_to_ucr_format', 'visualize_filter', 
                                 'viz_for_survey_paper', 'viz_cam', 'generate_results'], required=True)
    parser.add_argument('-t', '--time', type=int, required=True)
    parser.add_argument('-tp','--cell_types', type=str, required=True)
    parser.add_argument('-g','--gradient', type=bool, required=False, default=False)
    args = parser.parse_args()
    args.cell_types = sorted(args.cell_types.split('-'))
    print(args.cell_types)
    run(args)
