import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

from evaluation.log import log
from evaluation.load import get_max_acc_experiment, get_predictions

@log
def generate_loss_acc_plot(filename, experiment):
    fig, ax = plt.subplots(2,2, figsize=(15,8))
    plt.suptitle(experiment['name'])
    ax[0,0].set_title("Trn loss")
    ax[0,1].set_title("Val loss")
    ax[1,0].set_title("Trn Accuracy")
    ax[1,1].set_title("Val Accuracy")
    for n, exp in enumerate(experiment['experiments']):
        exp_hist = pd.read_csv(os.path.join(exp, 'history.csv'))
        ax[0,0].plot(exp_hist.loss, label=n)
        ax[0,1].plot(exp_hist.val_loss, label=n)
        ax[1,0].plot(exp_hist.accuracy, label=n)
        ax[1,1].plot(exp_hist.val_accuracy, label=n)
    ax[0,0].legend()
    ax[0,1].legend()
    ax[1,0].legend()
    ax[1,1].legend()
    ax[1,0].set_ylim((0,1))
    ax[1,1].set_ylim((0,1))
    plt.savefig(filename)
    plt.close()
    
@log
def generate_tr_tst_plot(filename, experiments):   
    fig, ax = plt.subplots(1,2, figsize=(15,8))
    ax[0].set_title("Trn Accuracy")
    ax[1].set_title("Val Accuracy")
    for exp in experiments:
    #     if exp['name'] not in ['resnet', 'fcn']: 
        mx_idx = get_max_acc_experiment(exp)
        exp_hist = pd.read_csv(os.path.join(exp['experiments'][mx_idx], 'history.csv'))
        ax[0].plot(exp_hist.accuracy, label=exp['name'], alpha=.5)
        ax[1].plot(exp_hist.val_accuracy, label=exp['name'], alpha=.5)
    ax[0].legend()
    ax[1].legend()
    ax[0].set_xlim((0, 400))
    ax[1].set_xlim((0, 400))
    plt.savefig(filename)
    plt.close()

@log
def generate_tst_pred_plot(experiment, x_test, y_test, labels, cmap=None):
    label_count = len(labels)
    if cmap is None:
        cm = mpl.cm.get_cmap('Set1', label_count)
        cmap = [mpl.colors.rgb2hex(cm(i)) for i in range(label_count)]
    y_pred, _ = get_predictions(experiment, labels)
    fig, ax = plt.subplots(len(labels),1, figsize=(15,len(labels) * 10))
    fig.suptitle("Test predictions")
    for i in range(label_count):
        for j in range(label_count):
            ax[i].plot([], c=cmap[j], label=labels[j])
        ax[i].legend()
        ax[i].set_title(''.join([labels[i], '-predictions']))

    for n in range(x_test.shape[0]):
        ax[y_test[n]].plot(x_test[n,:], c=cmap[y_pred[n]])
    plt.savefig(os.path.join(experiment, 'tst-predictions-types.png'))
    plt.close()

@log
def generate_preds_plot(experiment, x_test, y_test, labels):
    label_count = len(labels)
    cmap = ['r', 'g']
    y_pred, _ = get_predictions(experiment, labels)
    fig, ax = plt.subplots(len(labels),1, figsize=(15,len(labels) * 10))
    for i in range(label_count):
        for n, label in enumerate(['false', 'true']):
            ax[i].plot([], c=cmap[n], label=label)
        ax[i].legend()
        ax[i].set_title(''.join([labels[i], '_predictions']))
            
    for m in range(x_test.shape[0]):
        ax[y_test[m]].plot(x_test[m, :], c=cmap[int(y_pred[m] == y_test[m])])
    plt.savefig(os.path.join(experiment, 'tst-predictions.png'))
    plt.close()

@log
def generate_test_hist_plot(filename, x_test, y_test, labels, label_ids, cmap=None):
    label_count = len(labels)
    if cmap is None:
        cm = mpl.cm.get_cmap('Set1', label_count)
        cmap = [mpl.colors.rgb2hex(cm(i)) for i in range(label_count)]
    x_test_lst = x_test[:, -1]
    separated_x_test = [ x_test[np.where(y_test == n)] for n in label_ids]
    bins_type = []
    for x_test_type in separated_x_test:
        bins, bin_edges = np.histogram(x_test_type[:, -1], bins=50, range=(min(x_test_lst), max(x_test_lst)),density=False)
        bins_type.append(bins)
    bins, bin_edges = np.histogram(x_test_lst, bins=50, range=(min(x_test_lst), max(x_test_lst)),density=False)
    plt.figure(figsize=(12,8))
    for m in label_ids:
        plt.bar(0, 0, width=0, color=cmap[m], label=labels[m])
    for n, (l_edge, r_edge) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        bottom = 0
        for m in label_ids:
            plt.bar(l_edge, bins_type[m][n], width=(r_edge - l_edge), color=cmap[m], edgecolor='black', linewidth=.2, bottom=bottom)
            bottom += bins_type[m][n]
    plt.legend()
    plt.savefig(filename)
    plt.close()

@log
def generate_test_type_hist_plot(experiment, x_test, y_test, labels, label_ids, cmap=None):
    label_count = len(labels)
    x_test_lst = x_test[:, -1]
    if cmap is None:
        cm = mpl.cm.get_cmap('Set1', label_count)
        cmap = [mpl.colors.rgb2hex(cm(i)) for i in range(label_count)]
    y_pred, _ = get_predictions(experiment, labels)
    fig, ax = plt.subplots(label_count, 1, figsize=(12, label_count * 8))
    fig.suptitle("Test predictions histogram")
    for i, (name, tag) in enumerate(zip(labels, label_ids)):
        label_ids_l = label_ids[i:] + label_ids[:i]
        label_slice = np.where(y_test == tag)
        x_test_type = x_test[label_slice]
        x_test_type_lst = x_test_type[:, -1]
        # y_test_type = y_test[label_slice]
        y_pred_type = y_pred[label_slice]
        bins_type = []
        for n in label_ids:
            bins, bin_edges = np.histogram(x_test_type_lst[np.where(y_pred_type == n)], bins=50, range=(min(x_test_lst), max(x_test_lst)),density=False)
            bins_type.append(bins)
        bins, bin_edges = np.histogram(x_test_type_lst, bins=50, range=(min(x_test_lst), max(x_test_lst)),density=False)
        ax[tag].set_title(' '.join([name, 'predictions']))
        for m in label_ids:
            container = ax[tag].bar(0, 0, width=0, color=cmap[m], label=labels[m])
        for n, (l_edge, r_edge) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
            bottom = 0
            for m in label_ids_l:
                container = ax[tag].bar(l_edge, bins_type[m][n], width=(r_edge - l_edge), color=cmap[m], edgecolor='black', linewidth=.2, bottom=bottom)
                bottom += bins_type[m][n]
        ax[tag].legend()
    plt.savefig(os.path.join(experiment, 'tst-types-hist.png'))
    plt.close()
    
@log
def generate_conf_matrix(experiment, test_labels, labels):
    y_pred, pred_labels = get_predictions(experiment, labels)
    cm = confusion_matrix(test_labels, pred_labels, labels=labels, normalize='true')
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    disp.plot(cmap=mpl.cm.Blues)
    plt.savefig(os.path.join(experiment, 'conf-matrix.png'))
    plt.close()
    
@log
def generate_conf_graph(experiment, test_labels, labels, label_ids, cmap=None):
    _, pred_labels = get_predictions(experiment, labels)
    cm = confusion_matrix(test_labels, pred_labels, labels=labels, normalize='true')
    label_count = len(labels)
    if cmap is None:
        cm = mpl.cm.get_cmap('Set1', label_count)
        cmap = [mpl.colors.rgb2hex(cm(i)) for i in range(label_count)]
    radius = 1
    G = nx.DiGraph(edge_layout='curved')
    
    for i in label_ids:
        theta = 2 * np.pi * i / len(label_ids)
        if len(label_ids) > 2:
            theta += np.pi / 2
        G.add_node(
            label_ids[i],
            pos=((radius * np.cos(theta), radius * np.sin(theta))),
            color=cmap[i],
            weight=round(cm[i,i] * 7000),
                  )

    for i in label_ids:
        for j in label_ids:
            if i == j:
                continue
            G.add_edge(i,j,
                        label = labels[i] + ' to ' + labels[j],
                        color = cmap[i],
                        weight = cm[i, j] * 30,
                       )
            G.add_edge(j,i,
                        label = labels[i] + ' to ' + labels[j],
                        color = cmap[j],
                        weight = cm[j, i] * 30,
                       )

    edges = G.edges()
    pos = list(nx.get_node_attributes(G, 'pos').values())
    node_colors = list(nx.get_node_attributes(G, 'color').values())
    node_weights = list(nx.get_node_attributes(G, 'weight').values())
    edge_colors = list(nx.get_edge_attributes(G, 'color').values())
    edge_weights = list(nx.get_edge_attributes(G, 'weight').values())

    # Draw nodes and edges
    plt.figure(figsize=(8,8))
    nodes = nx.draw_networkx_nodes(
        G, pos, 
        node_size=node_weights, 
        node_color=node_colors,
    #     edgecolors='black'
    )
    edges = nx.draw_networkx_edges(
        G, pos,
        node_size=node_weights,
        edge_color=edge_colors,
        width=edge_weights,
        connectionstyle="arc3,rad=0.1",
        arrowstyle='-'
    )
    nx.draw_networkx_labels(
        G, pos, 
        labels={n: label for n, label in enumerate(labels)},

    )
    plt.gca().set_frame_on(False)
    plt.xlim((-1.4,1.4))
    plt.ylim((-1.4,1.4))
    plt.savefig(os.path.join(experiment, 'conf-graph.png'))
    plt.close()