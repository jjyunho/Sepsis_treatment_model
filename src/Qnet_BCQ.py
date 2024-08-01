import torch
import torch.nn as nn
import torch.nn.functional as F

class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(47, 1024)
        self.bn1 = nn.BatchNorm1d(num_features=1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.bn2 = nn.BatchNorm1d(num_features=1024)
        self.fc6 = nn.Linear(1024, 25)
        self.bn6 = nn.BatchNorm1d(num_features=25)

        self.fc2_ = nn.Linear(1024, 1024)
        self.bn2_ = nn.BatchNorm1d(num_features=1024)
        self.fc6_ = nn.Linear(1024, 1)
        self.bn6_ = nn.BatchNorm1d(num_features=1)

        ## BCQ ##

        self.fc2_i = nn.Linear(1024, 1024)
        self.bn2_i = nn.BatchNorm1d(num_features=1024)
        self.fc6_i = nn.Linear(1024, 25)
        self.bn6_i = nn.BatchNorm1d(num_features=25)

        self.fc2__i = nn.Linear(1024, 1024)
        self.bn2__i = nn.BatchNorm1d(num_features=1024)
        self.fc6__i = nn.Linear(1024, 25)
        self.bn6__i = nn.BatchNorm1d(num_features=25)

        ##################################################

        torch.nn.init.xavier_uniform_(self.fc1.weight)

        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.fc6.weight)
        torch.nn.init.xavier_uniform_(self.fc2_.weight)
        torch.nn.init.xavier_uniform_(self.fc6_.weight)

        torch.nn.init.xavier_uniform_(self.fc2_i.weight)
        torch.nn.init.xavier_uniform_(self.fc6_i.weight)
        torch.nn.init.xavier_uniform_(self.fc2__i.weight)
        torch.nn.init.xavier_uniform_(self.fc6__i.weight)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))

        adv = F.relu(self.bn2(self.fc2(x)))
        adv = (self.bn6(self.fc6(adv)))

        value = F.relu(self.bn2_(self.fc2_(x)))
        value = (self.bn6_(self.fc6_(value)))

        advAverage = torch.mean(adv, dim=1, keepdim=True)

        q = value + adv - advAverage

        # BCQ : https://github.com/sfujim/BCQ/blob/master/discrete_BCQ/discrete_BCQ.py  # line 29~31
        # adv_i = F.relu(self.bn2_i(self.fc2_i(x)))
        # adv_i = (self.bn6_i(self.fc6_i(adv_i)))

        value_i = F.relu(self.bn2__i(self.fc2__i(x)))
        value_i = (self.bn6__i(self.fc6__i(value_i)))

        # advAverage_i = torch.mean(adv_i, dim=1, keepdim=True)
        # i = value_i + adv_i - advAverage_i

        return q, F.log_softmax(value_i, dim=1), value_i

    def sample_action(self, obs):
        out = self.forward(obs)
        return torch.argmax(out, dim=1).item()