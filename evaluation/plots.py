import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import re
from iteround import saferound

from evaluation.log import log
from evaluation.load import get_max_acc_experiment, get_predictions
import warnings
warnings.filterwarnings("ignore")

@log
def generate_mean_plot(filename, x_data, y_data, labels, cmap, names, small=False):
    sz = (4, 3) if small else (6,5)
    fig, ax = plt.subplots(figsize=sz)
    for l, c, n in zip(labels, cmap, names):
        avg = np.mean(np.array(x_data[np.where(y_data == l)]), axis=0)
        ax.plot(avg, label=n, c=c, linewidth=3)
    ticks = [int(re.sub(u"\u2212", "-", i.get_text())) for i in ax.get_xticklabels()]
    ax.set_xticklabels([int(item * 9 / 60) for item in ticks])
    plt.xlabel("Time(min)", fontsize=12)
    plt.ylabel("WS(pm)", fontsize=12)
    plt.legend()
    plt.tight_layout()
    ext = f"(small)" if small else ""
    filename = filename.split('.')
    filename[0] += ext
    filename = '.'.join(filename)
    plt.savefig(filename, dpi=200)
    plt.close()

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
    plt.savefig(filename, dpi=200)
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
    plt.savefig(filename, dpi=200)
    plt.close()

@log
def generate_tst_pred_plot(experiment, x_test, y_test, labels, cmap=None, names=None):
    label_count = len(labels)
    if cmap is None:
        cm = mpl.cm.get_cmap('Set1', label_count)
        cmap = [mpl.colors.rgb2hex(cm(i)) for i in range(label_count)]
    y_pred, _ = get_predictions(experiment, labels)
    fig, ax = plt.subplots(len(labels),1, figsize=(5,len(labels) * 4))
    # fig.suptitle("Test predictions")
    for i in range(label_count):
        for j in range(label_count):
            label = names[j] if names is not None else labels[j]
            ax[i].plot([], c=cmap[j], label=label)
        ax[i].legend()
        label = names[i] if names is not None else labels[i]
        ax[i].set_title(label)
        ax[i].set_xlabel('Time(s)')
        ax[i].set_ylabel('WS(pm)')

    for n in range(x_test.shape[0]):
        ax[y_test[n]].plot(x_test[n,:], c=cmap[y_pred[n]])

    for i in range(label_count):
        ticks = [int(re.sub(u"\u2212", "-", i.get_text())) for i in ax[i].get_xticklabels()]
        ax[i].set_xticklabels([item * 3 for item in ticks])
    plt.tight_layout()
    plt.savefig(os.path.join(experiment, 'tst-predictions-types.png'), dpi=200)
    plt.close()

@log
def generate_preds_plot(experiment, x_test, y_test, labels, names=None):
    label_count = len(labels)
    cmap = ['r', 'g']
    y_pred, _ = get_predictions(experiment, labels)
    fig, ax = plt.subplots(len(labels),1, figsize=(5,len(labels) * 4))
    for i in range(label_count):
        for n, label in enumerate(['false', 'true']):
            ax[i].plot([], c=cmap[n], label=label)
        ax[i].legend()
        label = names[i] if names is not None else labels[i]
        ax[i].set_title(label)
        ax[i].set_xlabel('Time(s)')
        ax[i].set_ylabel('WS(pm)')

    for m in range(x_test.shape[0]):
        ax[y_test[m]].plot(x_test[m, :], c=cmap[int(y_pred[m] == y_test[m])])

    for i in range(label_count):
        ticks = [int(re.sub(u"\u2212", "-", i.get_text())) for i in ax[i].get_xticklabels()]
        ax[i].set_xticklabels([item * 3 for item in ticks])
    plt.tight_layout()
    plt.savefig(os.path.join(experiment, 'tst-predictions.png'), dpi=200)
    plt.close()

@log
def generate_test_hist_plot(filename, x_test, y_test, labels, label_ids, cmap=None, names=None, small=False, add_lines=False):
    sz = (4, 3) if small else (5,4)
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
    fig, ax = plt.subplots(figsize=sz)
    for m in label_ids:
        label = names[m] if names is not None else labels[m]
        ax.bar(0, 0, width=0, color=cmap[m], label=label)
    for n, (l_edge, r_edge) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        width = r_edge - l_edge
        bottom = 0
        for m in label_ids:
            ax.bar(l_edge + (width/2), bins_type[m][n], width=(r_edge - l_edge), color=cmap[m], edgecolor='black', linewidth=.2, bottom=bottom)
            bottom += bins_type[m][n]
    if add_lines:
        mx = max(bins) * 0.12
        dff = mx / (len(label_ids) + 2)
        for i in label_ids:
            sample = x_test[np.where(y_test == i), -1]
            if sample.shape[0] != 0:
                line = list(range(int(np.min(sample)), int(np.max(sample))))
                if len(line) != 0:
                    ax.plot(line, [-((i+1) * dff)] * len(line), color=cmap[i])
                    ax.scatter([min(line), max(line)], [-((i+1) * dff), -((i+1) * dff)], color=cmap[i], s=10)
                else:
                    ax.scatter([int(np.min(sample))], [-((m+1) * dff)], color=cmap[m], s=10)
        ax.set_ylim(-(max(bins) * .12), max(bins) * 1.05)
    width = np.max(x_test_lst) - np.min(x_test_lst)
    ax.set_xlim(np.min(x_test_lst) - .02 * width, np.max(x_test_lst) + .02 * width)
    ax.set_xlabel("WS(pm)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    plt.legend()
    plt.tight_layout()
    props = list(filter(lambda a: a != "", ['small' if small else '', 'ranges' if add_lines else '']))
    ext = f"({','.join(props)})" if len(props) > 0 else ""
    filename = filename.split('.')
    filename[0] += ext
    filename = '.'.join(filename)
    plt.savefig(filename, dpi=200)
    plt.close()

@log
def generate_test_type_hist_plot(experiment, x_test, y_test, labels, label_ids, cmap=None, names=None, small=False, add_lines=False):
    label_count = len(labels)
    sz = (5, label_count * 4)
    if small:
        sz = (4, label_count * 3)
    x_test_lst = x_test[:, -1]
    if cmap is None:
        cm = mpl.cm.get_cmap('Set1', label_count)
        cmap = [mpl.colors.rgb2hex(cm(i)) for i in range(label_count)]
    y_pred, _ = get_predictions(experiment, labels)
    fig, ax = plt.subplots(label_count, 1, figsize=sz)
    # fig.suptitle("Test predictions histogram")
    for i, (name, tag) in enumerate(zip(names, label_ids)):
        label_ids_l = label_ids[i:] + label_ids[:i]
        label_slice = np.where(y_test == tag)
        x_test_type = x_test[label_slice]
        x_test_type_lst = x_test_type[:, -1]
        # y_test_type = y_test[label_slice]
        y_pred_type = y_pred[label_slice]
        bins_type = []
        for n in label_ids:
            bins, bin_edges = np.histogram(x_test_type_lst[np.where(y_pred_type == n)], bins=50, range=(min(x_test_lst), max(x_test_lst)), density=False)
            bins_type.append(bins)
        bins, bin_edges = np.histogram(x_test_type_lst, bins=50, range=(min(x_test_lst), max(x_test_lst)), density=False)
        ax[tag].set_title(name, fontsize=12)
        for m in label_ids:
            label = names[m] if names is not None else labels[m]
            container = ax[tag].bar(0, 0, width=0, color=cmap[m], label=label)
        for n, (l_edge, r_edge) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
            width = r_edge - l_edge
            bottom = 0
            for m in label_ids_l:
                container = ax[tag].bar(l_edge + (width/2), bins_type[m][n], width=width, color=cmap[m], edgecolor='black', linewidth=.2, bottom=bottom)
                bottom += bins_type[m][n]
        if add_lines:
            mx = max(bins) * 0.12
            dff = mx / (len(label_ids) + 2)
            for m in label_ids:
                sample = x_test_type_lst[np.where(y_pred_type == m)]
                if sample.shape[0] != 0:
                    line = list(range(int(np.min(sample)), int(np.max(sample))))
                    if len(line) != 0:
                        ax[tag].plot(line, [-((m+1) * dff)] * len(line), color=cmap[m])
                        ax[tag].scatter([min(line), max(line)], [-((m+1) * dff), -((m+1) * dff)], color=cmap[m], s=10)
                    else:
                        ax[tag].scatter([int(np.min(sample))], [-((m+1) * dff)], color=cmap[m], s=10)
                        
            ax[tag].set_ylim(-(max(bins) * .12), max(bins) * 1.05)
        width = np.max(x_test_lst) - np.min(x_test_lst)
        ax[tag].set_xlim(np.min(x_test_lst) - .02 * width, np.max(x_test_lst) + .02 * width)
        ax[tag].legend()
        ax[tag].set_xlabel("WS(pm)", fontsize=12)
        ax[tag].set_ylabel("Count", fontsize=12)
    plt.tight_layout()
    props = list(filter(lambda a: a != "", ['small' if small else '', 'ranges' if add_lines else '']))
    ext = f"({','.join(props)})" if len(props) > 0 else ""
    plt.savefig(os.path.join(experiment, f'tst-types-hist{ext}.png'), dpi=300)
    plt.close()

@log
def generate_conf_matrix(experiment, test_labels, labels, names=None):
    y_pred, pred_labels = get_predictions(experiment, labels)
    cm = confusion_matrix(test_labels, pred_labels, labels=labels, normalize='true')
    cm = np.array([saferound(cm[i, :], 2) for i in range(cm.shape[0]) ])
    print(cm)
    disp = ConfusionMatrixDisplay(cm, display_labels=names if names is not None else labels)
    pl = disp.plot(cmap=mpl.cm.Blues, xticks_rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(experiment, 'conf-matrix.png'), pad_inches=5, dpi=300)
    plt.close()

@log
def generate_conf_graph(experiment, test_labels, labels, label_ids, cmap=None, names=None, small=False):
    sz = (10,10)
    node_ratio = 7000
    edge_ratio = 30
    if small:
        sz = (5,5)
        node_ratio = 1500
        edge_ratio = 20

    _, pred_labels = get_predictions(experiment, labels)
    cm = confusion_matrix(test_labels, pred_labels, labels=labels, normalize='true')
    label_count = len(labels)
    if cmap is None:
        cmap = mpl.cm.get_cmap('Set1', label_count)
        cmap = [mpl.colors.rgb2hex(cm(i)) for i in range(label_count)]
    radius = 1
    G = nx.DiGraph(edge_layout='curved')

    text_pos = []

    for i in label_ids:
        theta = 2 * np.pi * i / len(label_ids)
        if len(label_ids) > 2:
            theta += np.pi / 2
        G.add_node(
            label_ids[i],
            pos=((radius * np.cos(theta), radius * np.sin(theta))),
            color=cmap[i],
            weight=round(cm[i,i] * node_ratio),
                  )
        if theta % (2*np.pi) >= 0 and theta % (2*np.pi) <= np.pi:
            text_pos.append((radius * np.cos(theta), radius * np.sin(theta) + .3))
        else:
            text_pos.append((radius * np.cos(theta), radius * np.sin(theta) - .3))

    for i in label_ids:
        for j in label_ids:
            if i == j:
                continue
            G.add_edge(i,j,
                        label = labels[i] + ' to ' + labels[j],
                        color = cmap[i],
                        weight = cm[i, j] * edge_ratio,
                       )
            G.add_edge(j,i,
                        label = labels[i] + ' to ' + labels[j],
                        color = cmap[j],
                        weight = cm[j, i] * edge_ratio,
                       )

    edges = G.edges()
    pos = list(nx.get_node_attributes(G, 'pos').values())
    node_colors = list(nx.get_node_attributes(G, 'color').values())
    node_weights = list(nx.get_node_attributes(G, 'weight').values())
    edge_colors = list(nx.get_edge_attributes(G, 'color').values())
    edge_weights = list(nx.get_edge_attributes(G, 'weight').values())


    # Draw nodes and edges
    plt.figure(figsize=sz)
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
        G, text_pos,
        labels={n: label for n, label in enumerate(names if names is not None else labels)},

    )
    plt.gca().set_frame_on(False)
    plt.xlim((-1.6,1.6))
    plt.ylim((-1.6,1.6))
    plt.savefig(os.path.join(experiment, f'conf-graph{"(small)" if small else ""}.png'), dpi=200)
    plt.tight_layout()
    plt.close()