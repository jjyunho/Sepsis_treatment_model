import pickle
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import random

def read_results(num, id, epoch, result_path):
    results = dict()
    data = []
    # data = np.asarray(pickle.load(open(result_path + id + '/{}_{}_sample_trace_list.pickle'.format(1, epoch), 'rb')))
    # for i in range(num):
    #     with open(result_path + id + '/{}_{}_sample_trace_test_list.pickle'.format(i+1, epoch), 'rb') as f:
    #         data = data + pickle.load(f)
    with open(result_path + id + '/{}_{}_sample_trace_train_list.pickle'.format(1, epoch), 'rb') as f:
        data = data + pickle.load(f)

    _, num_col = data[0].size()
    arr = np.zeros((1, num_col))

    for i in range(len(data)):
        arr = np.concatenate((arr, data[i].numpy()), axis=0)
    arr = np.delete(arr, 0, axis=0)

    arr_df = pd.DataFrame(arr)
    arr_df = arr_df.groupby([0, 1]).mean()
    survival_data = arr_df.loc[arr_df[4] == 0]
    death_data = arr_df.loc[arr_df[4] == 1]

    survival_data = survival_data.reset_index()
    death_data = death_data.reset_index()

    sur_noise = np.random.uniform(low=-0.5, high=0.5, size=(len(survival_data),))
    death_noise = np.random.uniform(low=-0.5, high=0.5, size=(len(death_data),))

    survival_data[1] = survival_data[1] + sur_noise
    death_data[1] = death_data[1] + death_noise


    return survival_data, death_data


def plot_result(survival_data, death_data, legends, colors, epoch, result_path, dpi=300, figsize=(5, 8)):
    plt.clf()
    survival_x = survival_data[1].tolist()
    survival_y = survival_data[3].tolist() # data[3]: physician's q , data[2]: rein q
    death_x = death_data[1].tolist()
    death_y = death_data[3].tolist()
    plt.scatter(survival_x, survival_y, s=0.1, c="blue")
    plt.scatter(death_x, death_y, s=0.1, c="red")
    plt.xlabel("sequence")
    plt.ylabel("q_value")
    plt.show()
    plt.savefig('results_{}.png'.format(epoch))



def main():
    result_path = '../result/'  # 그래프를 그릴 결과 폴더가 모여있는 경로
    data_ids = [  # 그 안에 그리고 싶은 결과 폴더 이름
        'scene+Qdivisionhighlight_33_lr1e-05_lr-decay0.99_batch4096_epoch200_222099',

    ]
    legends = [
        'survive',
        'death',
    ]
    colors = [
        'blue',
        'red'
    ]

    data_set_name = 'tracing'  # 그래프 저장 시 뒤에 붙을 이름

    num_data = 1
    # epoch = [1, 5, 10, 20, 50, 100, 200, 300, 400, 500]
    epoch = [1, 5, 10, 20, 50, 100, 200]
    for i in epoch:
        survival_data, death_data = read_results(num_data, data_ids[0], str(i), result_path)
        plot_result(survival_data, death_data, legends, colors, str(i), result_path, dpi=300, figsize=(5, 8))

    # fig, axes = plt.subplots(nrows=2, ncols=1, sharey=True, dpi=300, figsize=(5, 8))
    # plot_result(data_ids, data_ids[0], axes[0], legends, colors, attr='Intervention Fluid', data=results)
    # plot_result(data_ids, data_ids[0], axes[1], legends, colors, attr='Vasopressor', data=results)


    # plt.tight_layout()
    # plt.savefig('results_{}.png'.format(data_set_name))
    # plt.gcf()


if __name__ == '__main__':
    main()