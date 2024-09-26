import pickle
import numpy as np
from matplotlib import pyplot as plt


def read_results(num, id, epoch, path):
    ai_ivfs = np.asarray([pickle.load(open(path + id + '/{}_AI_ivf_survival_rate_list.pickle'.format(1), 'rb')) for i in range(num)])
    ai_vasos = np.asarray([pickle.load(open(path + id + '/{}_AI_vaso_survival_rate_list.pickle'.format(1), 'rb')) for i in range(num)])
    physician = np.asarray([pickle.load(open(path + id + '/{}_physician_survival_rate_list.pickle'.format(1), 'rb')) for i in range(num)])

    return {'Intervention Fluid': ai_ivfs[:,:epoch], 'Vasopressor': ai_vasos[:,:epoch], 'physician': physician[:,:epoch]}


def plot_avg_with_std(ax, avg, std, label, color, style, alpha=0.2):
    ax.plot(np.arange(len(avg)), avg, linestyle=style, label=label, color=color)
    ax.fill_between(np.arange(len(avg)), avg - std, avg + std, alpha=alpha)


def plot_result_each(ax, coefficient, data, legend=False):
    avg_physician, std_physician = data['physician'].mean(axis=0), data['physician'].std(axis=0)
    avg_ai_ivfs, std_ai_ivfs = data['Intervention Fluid'].mean(axis=0), data['Intervention Fluid'].std(axis=0)
    avg_ai_vasos, std_ai_vasos = data['Vasopressor'].mean(axis=0), data['Vasopressor'].std(axis=0)

    ax.set_xlabel('Coef:{:>4}'.format(coefficient), fontsize=12)
    ax.xaxis.set_label_position('top')
    if legend: ax.set_ylabel('Survival rate')

    ax.plot(np.arange(len(avg_physician)), avg_physician, linestyle='--', label='Physician')
    ax.fill_between(np.arange(len(avg_physician)), avg_physician - std_physician, avg_physician + std_physician, alpha=0.2)

    ax.plot(np.arange(len(avg_ai_ivfs)), avg_ai_ivfs, linestyle='-', label='AI - Intervention fluid')
    ax.fill_between(np.arange(len(avg_ai_ivfs)), avg_ai_ivfs - std_ai_ivfs, avg_ai_ivfs + std_ai_ivfs, alpha=0.2)

    ax.plot(np.arange(len(avg_ai_vasos)), avg_ai_vasos, linestyle='-', label='AI - Vasopressor')
    ax.fill_between(np.arange(len(avg_ai_vasos)), avg_ai_vasos - std_ai_vasos, avg_ai_vasos + std_ai_vasos, alpha=0.2)

    if legend:
        ax.legend(loc='upper left')


def plot_result(data_ids, gt_ids, ax, legends, colors, attr, data, legend=True):
    avg_attr = dict()
    std_attr = dict()

    avg_gt, std_gt = data[gt_ids]['physician'].mean(axis=0), data[gt_ids]['physician'].std(axis=0)
    for data_id in data_ids:
        avg_attr[data_id], std_attr[data_id] = data[data_id][attr].mean(axis=0), data[data_id][attr].std(axis=0)

    ax.set_xlabel('{}'.format(attr), fontsize=12)
    ax.xaxis.set_label_position('bottom')
    ax.set_ylabel('Survival rate')

    plot_avg_with_std(ax, avg_gt, std_gt, 'Physician', 'blue', '--')
    for i in range(len(data_ids)):
        plot_avg_with_std(ax, avg_attr[data_ids[i]], std_attr[data_ids[i]], legends[i], colors[i], '-')

    if legend:
        ax.legend(loc='lower right', fontsize=8)


def get_data(id, epoch, num, path):
    ai_ivfs = np.asarray([pickle.load(open(path + id + '/{}_AI_ivf_survival_rate_list.pickle'.format(i + 1), 'rb')) for i in range(num)])
    ai_vasos = np.asarray([pickle.load(open(path + id + '/{}_AI_vaso_survival_rate_list.pickle'.format(i + 1), 'rb')) for i in range(num)])

    return {'Intervention Fluid': ai_ivfs[:, epoch], 'Vasopressor': ai_vasos[:, epoch]}


def main():
    result_path = '../result/'
    data_ids = [  # target folder
        # 'highlight_c100',
        # 'single_mimic2',
        'scene+Qdivisionhighlight_33_lr1e-05_lr-decay0.99_batch4096_epoch200_222099'
    ]
    legends = [
        'yh_single',
        # 'single',
    ]
    colors = [
        'red',
        # 'black'
    ]

    data_set_name = 'test'

    num_data = 1
    epoch = 500

    results = dict()
    for data_id in data_ids:
        results[data_id] = read_results(num_data, data_id, epoch, result_path)

    fig, axes = plt.subplots(nrows=2, ncols=1, sharey=True, dpi=300, figsize=(5, 8))

    plot_result(data_ids, data_ids[0], axes[0], legends, colors, attr='Intervention Fluid', data=results)
    plot_result(data_ids, data_ids[0], axes[1], legends, colors, attr='Vasopressor', data=results)

    plt.tight_layout()
    plt.savefig('results_{}.png'.format(data_set_name))
    plt.gcf()


if __name__ == '__main__':
    main()