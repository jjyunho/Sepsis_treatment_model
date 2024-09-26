import pickle
import numpy as np
import pandas as pd


def read_attribute_data(id, attribute, data_number, result_path):
    with open(result_path + id + '/{}_test_{}.pickle'.format(attribute, data_number), 'rb') as f:
        data = pickle.load(f)
    return np.asarray([np.asarray(x).reshape(-1) for x in data]).reshape(-1)


def factorize_action(ori_action):
    iv = [x // 5 for x in ori_action.reshape(-1)]
    vaso = [x % 5 for x in ori_action.reshape(-1)]

    return np.asarray(iv), np.asarray(vaso)


def calculate_deathrate_by_comp(dataframe, how):
    dataframe['diff_iv'] = abs(dataframe['ori_iv'] - dataframe['action_iv'])
    dataframe['diff_vaso'] = abs(dataframe['ori_vaso'] - dataframe['action_vaso'])
    df_by_id = dataframe.groupby('id').agg({'mortality': 'mean', 'diff_iv': 'sum', 'diff_vaso': 'sum'})

    partial_df = df_by_id.loc[df_by_id['diff_iv'] == 0]
    deathrate_iv_diff = partial_df['mortality'].mean()
    num_same_iv = len(partial_df)

    partial_df = df_by_id.loc[df_by_id['diff_vaso'] == 0]
    deathrate_vaso_diff = partial_df['mortality'].mean()
    num_same_vaso = len(partial_df)

    partial_df = df_by_id.loc[(df_by_id['diff_iv'] == 0) & (df_by_id['diff_vaso'] == 0)]
    deathrate_total_diff = partial_df['mortality'].mean()
    num_same_total = len(partial_df)

    return num_same_iv, deathrate_iv_diff, num_same_vaso, deathrate_vaso_diff, num_same_total, deathrate_total_diff


def get_deathrate_iv_vaso(data_id, data_num, how, path):
    id = read_attribute_data(data_id, 'id', data_num, path)
    action_iv, action_vaso = factorize_action(read_attribute_data(data_id, 'action', data_num, path))
    ori_iv, ori_vaso = factorize_action(read_attribute_data(data_id, 'ori_action', data_num, path))
    mortality = read_attribute_data(data_id, 'mortality', data_num, path)

    dataframe = pd.DataFrame({'id': id,
                              'action_iv': action_iv, 'action_vaso': action_vaso,
                              'ori_iv': ori_iv, 'ori_vaso': ori_vaso,
                              'mortality': mortality})

    return calculate_deathrate_by_comp(dataframe, how=how)


def main():
    result_path = './test/'
    data_ids = [  # folder name
        'yh_single',
        'sm_single',
    ]

    how = 'mean'
    num_data = 100

    results = dict()
    num_results = dict()
    for data_id in data_ids:
        results_iv_deathrate = []
        results_vaso_deathrate = []
        results_total_deathrate = []
        iv = []
        vaso = []
        total = []

        for i in range(num_data):
            num_iv, deathrate_iv, num_vaso, deathrate_vaso, num_total, deathrate_total = get_deathrate_iv_vaso(data_id, i + 1, how, result_path)
            results_iv_deathrate.append(deathrate_iv)
            iv.append(num_iv)
            results_vaso_deathrate.append(deathrate_vaso)
            vaso.append(num_vaso)
            results_total_deathrate.append(deathrate_total)
            total.append(num_total)

        avg_interval_deathrate_iv = np.asarray(results_iv_deathrate).mean(axis=0)
        avg_iv = np.asarray(iv).mean(axis=0)
        avg_interval_deathrate_vaso = np.asarray(results_vaso_deathrate).mean(axis=0)
        avg_vaso = np.asarray(vaso).mean(axis=0)
        avg_interval_deathrate_total = np.asarray(results_total_deathrate).mean(axis=0)
        avg_total = np.asarray(total).mean(axis=0)

        results[data_id] = {'iv': avg_interval_deathrate_iv, 'vaso': avg_interval_deathrate_vaso, 'total': avg_interval_deathrate_total}
        num_results[data_id] = {'iv': avg_iv, 'vaso': avg_vaso, 'total': avg_total}

    print("death rate", results)
    print("num", num_results)


if __name__ == '__main__':
    main()