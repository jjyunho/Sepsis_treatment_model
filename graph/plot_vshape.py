import pickle
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

interval_iv = np.asarray(range(-550, 550 + 1, 100))
interval_iv_med = np.asarray(range(-500, 500+1, 100))
interval_vaso = np.asarray(range(-105, 105 + 1, 10)) / 100
interval_vaso_med = np.asarray(range(-100, 100 + 1, 10)) / 100



def read_attribute_data(id, attribute, data_number, result_path):
    with open(result_path + id + '/{}_test_{}.pickle'.format(attribute, data_number), 'rb') as f:
        data = pickle.load(f)
    return np.asarray([np.asarray(x).reshape(-1) for x in data]).reshape(-1)


def factorize_action(ori_action):
    med_iv = [0.0, 30.0, 80.004, 290.0, 874.600433333333]
    med_vaso = [0.0, 0.04, 0.135, 0.27, 0.787]

    iv = [med_iv[x // 5] for x in ori_action.reshape(-1)]
    vaso = [med_vaso[int(np.floor(x % 5))] for x in ori_action.reshape(-1)]

    return np.asarray(iv), np.asarray(vaso)


def calculate_deathrate_by_diff(dataframe, how):
    dataframe['diff_iv'] = dataframe['ori_iv'] - dataframe['action_iv']
    dataframe['diff_vaso'] = dataframe['ori_vaso'] - dataframe['action_vaso']
    df_by_id = dataframe.groupby('id').agg({'mortality': 'mean',
                                            'diff_iv': how,
                                            'diff_vaso': how})

    deathrate_iv_diff = np.zeros(len(interval_iv)-1)
    deathrate_vaso_diff = np.zeros(len(interval_vaso)-1)

    for i in range(len(deathrate_iv_diff)):
        partial_df = df_by_id.loc[(interval_iv[i] < df_by_id['diff_iv']) & (df_by_id['diff_iv'] <= interval_iv[i + 1])]
        deathrate_iv_diff[i] = partial_df['mortality'].mean()

    for i in range(len(deathrate_vaso_diff)):
        partial_df = df_by_id.loc[(interval_vaso[i] < df_by_id['diff_vaso']) & (df_by_id['diff_vaso'] <= interval_vaso[i + 1])]
        deathrate_vaso_diff[i] = partial_df['mortality'].mean()

    return deathrate_iv_diff, deathrate_vaso_diff


def get_deathrate_iv_vaso(data_id, data_num, how, path):
    id = read_attribute_data(data_id, 'id', data_num, path)
    action_iv, action_vaso = factorize_action(read_attribute_data(data_id, 'action', data_num, path))
    ori_iv, ori_vaso = factorize_action(read_attribute_data(data_id, 'ori_action', data_num, path))
    mortality = read_attribute_data(data_id, 'mortality', data_num, path)

    dataframe = pd.DataFrame({'id': id,
                              'action_iv': action_iv, 'action_vaso': action_vaso,
                              # 'ori_iv': ori_iv, 'ori_vaso': ori_vaso,
                              'ori_iv': ori_iv, 'ori_vaso': ori_vaso,
                              'mortality': mortality})

    return calculate_deathrate_by_diff(dataframe, how=how)


def plot_vshape(axes, results, data_id, legend, i, color, is_value):
    axes[0].axvline(x=0, color='r', linestyle='--', linewidth = 0.5)
    axes[0].plot(interval_iv_med, results[data_id]['iv'], label=legend, color=color)
    axes[0].set_xlim((interval_iv_med.min() - 100, interval_iv_med.max() + 100))
    axes[0].set_xlabel('Intervention Fluid Difference')
    axes[0].set_ylabel('Death Rate')

    axes[1].axvline(x=0, color='r', linestyle='--', linewidth = 0.5)
    axes[1].plot(interval_vaso_med, results[data_id]['vaso'], label=legend, color=color)
    axes[1].set_xlim((interval_vaso_med.min() - 0.1, interval_vaso_med.max() + 0.1))
    axes[1].set_xlabel('Vasopressor Difference')
    axes[1].set_ylabel('Death Rate')

    if is_value:
        axes[0].text(interval_iv_med[8], 0.2+i*0.05, legend + ' ' + str(round(results[data_id]['iv'][5], 4)), fontsize=7, color=color)
        axes[1].text(interval_vaso_med[17], 0.2+i*0.1, legend + ' ' + str(round(results[data_id]['vaso'][10], 4)), fontsize=7, color=color)

    axes[0].legend(loc='upper left', fontsize=8)
    axes[1].legend(loc='upper left', fontsize=8)


def main():
    result_path = '../result/'
    data_ids = [  # target folder name
        'scene+Qdivisionhighlight_33_lr1e-05_lr-decay0.99_batch4096_epoch200_222099'

    ]
    legends = [
        # '100-6',
        # '10-6',
        # '1-6',
        # '10-7',
        # '1-7'
        'c_30_best'
    ]
    colors = [
        'red',
        # 'black',
        # 'yellow',
        # 'orange',
        # 'blue'
    ]

    data_set_name = 'test'  # graph name

    how = 'mean' #mean, sum
    num_data = 1

    is_value = 1

    results = dict()
    for data_id in data_ids:
        results_iv_deathrate = []
        results_vaso_deathrate = []

        # for i in range(num_data):
        #     deathrate_iv, deathrate_vaso = get_deathrate_iv_vaso(data_id, i + 1, how, result_path)
        #     results_iv_deathrate.append(deathrate_iv)
        #     results_vaso_deathrate.append(deathrate_vaso)
        deathrate_iv, deathrate_vaso = get_deathrate_iv_vaso(data_id, 1, how, result_path)
        results_iv_deathrate.append(deathrate_iv)
        results_vaso_deathrate.append(deathrate_vaso)

        avg_interval_deathrate_iv = np.asarray(results_iv_deathrate).mean(axis=0)
        avg_interval_deathrate_vaso = np.asarray(results_vaso_deathrate).mean(axis=0)

        results[data_id] = {'iv': avg_interval_deathrate_iv, 'vaso': avg_interval_deathrate_vaso}

    fig, axes = plt.subplots(nrows=2, ncols=1, dpi=300, figsize=(5, 8))

    for i in range(len(data_ids)):
        plot_vshape(axes, results, data_ids[i], legends[i], i, colors[i], is_value)

    plt.tight_layout()
    plt.savefig('v_shape_results_{}_{}.png'.format(data_set_name, is_value))
    plt.gcf()


if __name__ == '__main__':
    main()