import os
from evaluation.load import get_dictionary, get_test, get_colors, get_data
from evaluation.plots import *
from utils.utils import viz_cam
from data.load import load_dataset
import argparse

def run(data_path, result_path, color_path):
    print(data_path, result_path, color_path)
    result_paths = [ { 'name': name, 'path': os.path.join(result_path, name)} 
                    for name in os.listdir(result_path) if os.path.isdir(os.path.join(result_path, name))]
    experiments = []
    for result in result_paths:
        obj = result.copy()
        obj['experiments'] = sorted([os.path.join(result['path'], name) 
                                            for name in os.listdir(result['path']) if os.path.isdir(os.path.join(result['path'], name))])
        if len(obj['experiments']) == 0:
            obj['experiments'] = [result['path']]
        experiments.append(obj)

    labels, label_ids = get_dictionary(data_path)
    
    print(labels, label_ids)
    x_data, y_data, data_labels = get_data(data_path, labels)
    x_test, y_test, test_labels = get_test(data_path, labels)
    cmap, names = get_colors(color_path, labels)
    
    print("Generating average plot")
    generate_mean_plot(os.path.join(result_path, 'signal-means.png'), x_data, y_data, label_ids, cmap, names)
    generate_mean_plot(os.path.join(result_path, 'signal-means(small).png'), x_data, y_data, label_ids, cmap, names, True)

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
    generate_test_hist_plot(os.path.join(result_path, 'tst-hist(small).png'), x_test, y_test, labels, label_ids, cmap, names, True)

    print("Generating test types histogram plots")
    for experiment in experiments:
        for exp in experiment['experiments']:
            generate_test_type_hist_plot(exp, x_test, y_test, labels, label_ids, cmap, names)
            generate_test_type_hist_plot(exp, x_test, y_test, labels, label_ids, cmap, names, True)

    print("Generating confusion matrices")
    for experiment in experiments:
        for exp in experiment['experiments']:
            generate_conf_matrix(exp, test_labels, labels, names)

    print("Generating confusion graphs")
    for experiment in experiments:
        for exp in experiment['experiments']:
            generate_conf_graph(exp, test_labels, labels, label_ids, cmap, names)
            generate_conf_graph(exp, test_labels, labels, label_ids, cmap, names, True)

    for experiment in experiments:
        for exp in experiment['experiments']:
            if(any(classifier in exp for classifier in ['fcn', 'resnet', 'inception'])):
                dataset = load_dataset(data_path, resample=False, scale=False)
                viz_cam(dataset, exp)
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DL-4-TS-EVAL')
    parser.add_argument('-p', '--src_path', type=str, help="path to the data source folder", required=True)
    parser.add_argument('-d', '--dst_path', type=str, help="path to the result folder", required=True)
    parser.add_argument('-c', '--clr_path', type=str, required=False)
    args = parser.parse_args()
    run(args.src_path, args.dst_path, args.clr_path)