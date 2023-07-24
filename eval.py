import os
from evaluation.load import get_dictionary, get_test, get_colors, get_data
from evaluation.plots import *
from utils.utils import viz_cam
import argparse

def run(data_path, result_path, color_path):
    print(data_path, result_path, color_path)
    result_paths = [ { 'name': name, 'path': os.path.join(result_path, name)} 
                    for name in os.listdir(result_path) if os.path.isdir(os.path.join(result_path, name))]
    experiments = [ {'name': result['name'],'path': result['path'], 
                     'experiments': sorted([os.path.join(result['path'], name) 
                                            for name in os.listdir(result['path']) if os.path.isdir(os.path.join(result['path'], name))])} for result in result_paths]

    labels, label_ids = get_dictionary(data_path)
    x_data, y_data, data_labels = get_data(data_path, labels)
    x_test, y_test, test_labels = get_test(data_path, labels)
    cmap, names = get_colors(color_path, labels)
    
    print("Generating average plot")
    generate_mean_plot(os.path.join(result_path, 'signal-means.png'), x_data, y_data, label_ids, cmap, names)

    print("Generating loss-accuracy plots")
    for experiment in experiments:
        generate_loss_acc_plot(os.path.join(experiment['path'], experiment['name']+'-tr-tst-metrics.png'), experiment)

    print("Generating total train-test metrics plot")
    generate_tr_tst_plot(os.path.join(result_path, 'tr-tst-metrics.png'), experiments)

    print("Generating test prediction types plots")
    for experiment in experiments:
        for exp in experiment['experiments']:
            generate_tst_pred_plot(exp, x_test, y_test, labels, cmap, names)

    print("Generating test prediction plots")
    for experiment in experiments:
        for exp in experiment['experiments']:
            generate_preds_plot(exp, x_test, y_test, labels, names)

    print("Generating test histogram plots")
    generate_test_hist_plot(os.path.join(result_path, 'tst-hist.png'), x_test, y_test, labels, label_ids, cmap, names)

    print("Generating test types histogram plots")
    for experiment in experiments:
        for exp in experiment['experiments']:
            generate_test_type_hist_plot(exp, x_test, y_test, labels, label_ids, cmap, names)

    print("Generating confusion matrices")
    for experiment in experiments:
        for exp in experiment['experiments']:
            generate_conf_matrix(exp, test_labels, labels, names)

    print("Generating confusion graphs")
    for experiment in experiments:
        for exp in experiment['experiments']:
            generate_conf_graph(exp, test_labels, labels, label_ids, cmap, names)

    for experiment in experiments:
        for exp in experiment['experiments']:
            if(any(classifier in exp for classifier in ['fcn', 'resnet', 'inception'])):
                viz_cam(data_path, exp, names)
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DL-4-TS-EVAL')
    parser.add_argument('-p', '--src_path', type=str, help="path to the data source folder", required=True)
    parser.add_argument('-d', '--dst_path', type=str, help="path to the result folder", required=True)
    parser.add_argument('-c', '--clr_path', type=str, required=False)
    args = parser.parse_args()
    run(args.src_path, args.dst_path, args.clr_path)