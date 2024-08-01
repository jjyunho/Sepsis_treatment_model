import os
import numpy as np

from scipy import io


# def load_mat_files(file_name='../data/MIMICzs_SASR_mortal3', key_name='MIMICzs_SASR'):
def load_mat_files(file_name='../data/MIMICraw_SASR2', key_name='MIMICraw_SASR2'):
    matrix = io.loadmat(os.path.join(file_name+'.mat'))

    if key_name is not False:
        matrix = matrix[key_name]

    return matrix


def data_norm(SASR, args):
    for i in range(96):
        if i == 47 or i == 95:
            pass
        else:
            SASR[:, i][SASR[:, i] >= args.clip] = args.clip
            SASR[:, i][SASR[:, i] <= -args.clip] = -args.clip
            max_value = np.max(SASR[:, i], axis=0)
            min_value = np.min(SASR[:, i], axis=0)
            SASR[:, i] = ((SASR[:, i] - min_value) / (max_value - min_value))
    return SASR


def split_episodes(SASR):
    n, m = SASR.shape
    episodes = []
    episodes.append([])
    num_patient = 0
    for i in range(n):
        episodes[num_patient].append(SASR[i, :])
        if SASR[i, 95] == 100 or SASR[i, 95] == -100:
            num_patient += 1
            episodes.append([])
    return episodes


def get_data_from_frame(frame):
    return (
        frame[0:47],                        # s
        frame[47] - 1,                      # a
        frame[95] / 100,                    # r
        frame[48:95],                       # s_prime
        0.0 if frame[95] != 0.0 else 1.0,   # done_mask
        frame[96],                          # death
        frame[97],                          # iv
        frame[98],                          # vaso
        frame[99],                          # id
        frame[100]                          # sequence(inverse)
    )


def split_idxes(method, n_rank, episode, split_size, seed=1337, shuffle=True):
    episode_idxes = np.arange(len(episode))
    np.random.seed(seed)
    if shuffle:
        np.random.shuffle(episode_idxes)

    train_idxes = []
    valid_idxes = []
    test_idxes = []

    train_len = int(split_size[0] * len(episode))
    val_len = int((split_size[0] + split_size[1]) * len(episode))


    if method == "single":
        train_idxes = episode_idxes[:train_len]
        valid_idxes = episode_idxes[train_len:val_len]
        test_idxes = episode_idxes[val_len:]


    return list(train_idxes), list(valid_idxes), list(test_idxes)
