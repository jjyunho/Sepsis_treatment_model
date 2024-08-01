from collections import deque
import itertools
import random
import torch


class ReplayBuffer():
    def __init__(self, device, n_rank):
        self.device = device

        self.dqn_train_buffer = deque(maxlen=223000)
        self.dqn_val_buffer = deque(maxlen=56000)

        self.dqn_test_buffer = deque(maxlen=223000)

    def dqn_train_put(self, transition):
        self.dqn_train_buffer.append(transition)


    def dqn_val_put(self, transition):
        self.dqn_val_buffer.append(transition)


    def dqn_test_put(self, transition):
        self.dqn_test_buffer.append(transition)

    def sample(self, epi, sample_num, rank):
        if rank == 1: mini_batch = random.sample(self.dqn_train_buffer, sample_num)
        elif rank == 99: mini_batch = list(itertools.islice(self.dqn_val_buffer, 0+(epi*(sample_num+1)), sample_num+(epi*(sample_num+1))))
        elif rank == 100: mini_batch = list(itertools.islice(self.dqn_train_buffer, 0+(epi*(sample_num+1)), sample_num+(epi*(sample_num+1))))
        elif rank == 10: mini_batch = list(itertools.islice(self.dqn_test_buffer, 0+(epi*(sample_num+1)), sample_num+(epi*(sample_num+1))))

        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst, death_lst, iv_lst, vaso_lst, id_lst, sq_lst = [], [], [], [], [], [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask, death, iv, vaso, id, sq = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])
            death_lst.append([death])
            iv_lst.append([iv])
            vaso_lst.append([vaso])
            id_lst.append([id])
            sq_lst.append([sq])

        return torch.tensor(s_lst, dtype=torch.float).to(self.device), \
               torch.tensor(a_lst, dtype=torch.long).to(self.device), \
               torch.tensor(r_lst).to(self.device).float(), \
               torch.tensor(s_prime_lst, dtype=torch.float).to(self.device), \
               torch.tensor(done_mask_lst).to(self.device), \
               torch.tensor(death_lst, dtype=torch.float).to(self.device), \
               torch.tensor(iv_lst).to(self.device), \
               torch.tensor(vaso_lst).to(self.device), \
               torch.tensor(id_lst).to(self.device), \
               torch.tensor(sq_lst).to(self.device)

    def size(self):
        return len(self.dqn_test_buffer)


    def test_size(self):
        return len(self.dqn_test_buffer)

    def val_size(self):
        return len(self.dqn_val_buffer)