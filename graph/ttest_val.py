import pickle
import numpy as np
from scipy import stats


def get_data(result_path, id, epoch, num):
    ai_ivfs = np.asarray([pickle.load(open(result_path + id + '/{}_AI_ivf_survival_rate_list.pickle'.format(i + 1), 'rb')) for i in range(num)])
    ai_vasos = np.asarray([pickle.load(open(result_path + id + '/{}_AI_vaso_survival_rate_list.pickle'.format(i + 1), 'rb')) for i in range(num)])

    return {'Intervention Fluid': ai_ivfs[:, epoch], 'Vasopressor': ai_vasos[:, epoch]}


def get_p_value(a, b):
    t, p = stats.ttest_ind(a, b)
    return p


def main():
    result_path = './test/'  # 그래프를 그릴 결과 폴더가 모여있는 경로
    data_ids = [  # 그 안에 그리고 싶은 결과 폴더 이름
        'highlight_c100',
        'single_mimic2',
    ]

    num_data = 100
    num_epoch = 10  # t test에 사용할 epoch 수 (뒤에서부터)
    epoch = 500

    datas = dict()
    ps_ivf = 0
    ps_vaso = 0

    for i in range(num_epoch):
        for data_id in data_ids:
            datas[data_id] = get_data(result_path, data_id, epoch-1-i, num_data)
        ps_ivf += (get_p_value(datas[data_ids[0]]['Intervention Fluid'], datas[data_ids[1]]['Intervention Fluid']))
        ps_vaso += (get_p_value(datas[data_ids[0]]['Vasopressor'], datas[data_ids[1]]['Vasopressor']))
    p_ivf = ps_ivf/num_epoch
    p_vaso = ps_vaso/num_epoch
    print(p_ivf)
    print(p_vaso)


if __name__ == '__main__':
    main()