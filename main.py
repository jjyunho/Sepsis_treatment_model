import argparse

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp

import itertools
import time
import pickle
from tqdm import tqdm
import pandas as pd
from matplotlib import pyplot as plt

# from Qnet import Qnet
from src.ReplayBuffer import ReplayBuffer
from src.utils import *

#########################################
# Hyperparameters #######################
#########################################
data_path = "./data/MIMIC_final_push"  # ../data/MIMICzs_SASR_mortal3, eICUZS_SASR4, MIMIC3
data_name = "MIMIC"  # MIMICzs_SASR2, eICUZS_SASR4, MIMIC

split_method = "single"
n_rank = 1
train_rank = [1]
stop_all = 1  # 프로세스 하나 끝나면 모든 프로세스 종료

split_size = [0.75, 0.1, 0.15, 0.0]  # train, valid, test, share
SPLIT_SEED = 1337

#########################################
#########################################

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

# 상수
ma1 = \
[0, 0.0400000000000000, 0.135000000000000, 0.270000000000000, 0.787000000000000,
0, 0.0400000000000000, 0.135000000000000, 0.270000000000000, 0.787000000000000,
0, 0.0400000000000000,0.135000000000000,0.270000000000000,0.787000000000000,
0,0.0400000000000000,0.135000000000000,0.270000000000000,0.787000000000000,
0,0.0400000000000000,0.135000000000000,0.270000000000000,0.787000000000000]

ma2 = \
[0,0,0,0,0,
30,30,30,30,30,
80.0040000000000,80.0040000000000,80.0040000000000,80.0040000000000,80.0040000000000,
290,290,290,290,290,
874.600433333333,874.600433333333,874.600433333333,874.600433333333,874.600433333333]



ma1_pd = pd.DataFrame(ma1)
ma1_pd.columns = ['vaso']
ma2_pd = pd.DataFrame(ma2)
ma2_pd.columns = ['iv']

key = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
key_pd = pd.DataFrame(key)
key_pd.columns = ['key']

med = pd.concat([key_pd, ma1_pd, ma2_pd], axis=1)
dict_med = med.set_index('key').T.to_dict('list')

def makedirs(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise

def train(global_model, memory, train_length, sequence, args, log_save_path):
    local_model = Qnet().to(device)
    local_model.load_state_dict(global_model.state_dict())

    loss_list = []
    value_list = []
    rein_q_list = []
    x = []
    physician_survival_rate_list = []
    AI_ivf_survival_rate_list = []
    AI_vaso_survival_rate_list = []

    for epoch in range(args.max_epoch):
        loss_sum = 0
        value_sum = 0
        rein_q_sum = 0

        for sample in range(train_length):
            for i in range(len(train_rank)):
                rank = train_rank[i]
                learning_rate = args.lr * args.lr_decay ** epoch
                optimizer = optim.Adam(global_model.parameters(), lr=learning_rate)

                s, a, r, s_prime, done_mask, death, iv, vaso, id, sq = memory.sample(1, args.batch_size, rank)

                if args.scene == "bcq":
                    q_out, imt, i = global_model.forward(s)
                    q_a = q_out.gather(1, a)
                    imt = imt.exp()  # torch.Size([32, 6])
                    imt = (imt / imt.max(1, keepdim=True)[0] > args.th).float()
                    rein_a = (imt * q_out + (1 - imt) * -1e8).argmax(1, keepdim=True)
                    rein_a = torch.reshape(rein_a, [args.batch_size, 1])
                    rein_q = q_out.gather(1, rein_a)
                    max_q_prime, imt_prime, i_prime = global_model.forward(s_prime)
                    imt_prime = imt_prime.exp()  # torch.Size([32, 6])
                    imt_prime = (imt_prime / imt_prime.max(1, keepdim=True)[0] > args.th).float()  # torch.Size([32, 6])
                    a_prime = (imt_prime * max_q_prime + (1 - imt_prime) * -1e8).argmax(1, keepdim=True)
                    max_q_prime = local_model.forward(s_prime)[0].gather(1, a_prime)
                else:
                    q_out = global_model(s)
                    q_a = q_out.gather(1, a)
                    rein_a = torch.argmax(global_model.forward(s), dim=1)
                    rein_a = torch.reshape(rein_a, [args.batch_size, 1])
                    rein_q = q_out.gather(1, rein_a)
                    a_prime = torch.argmax(global_model.forward(s_prime), dim=1)
                    a_prime = torch.reshape(a_prime, [args.batch_size, 1])
                    max_q_prime = local_model.forward(s_prime).gather(1, a_prime)

                if float(torch.max(rein_q)) > args.highlight_coefficient and args.search == True:
                    args.isover = True
                    args.highlight_coefficient = int(torch.max(rein_q)) + 2
                    return 0


                ##loss
                if args.scene == "highlight":
                    target = (args.highlight_coefficient * r) + args.gamma * max_q_prime * done_mask
                    loss = F.mse_loss(q_a, target)/(args.highlight_coefficient ** 2)
                elif args.scene == "highreward":
                    target = (args.highlight_coefficient * r) + args.gamma * max_q_prime * done_mask
                    loss = F.mse_loss(q_a, target)
                elif args.scene == "cql":
                    target = r + args.gamma * max_q_prime * done_mask
                    logsumexp = torch.logsumexp(q_out, dim=1, keepdim=True)
                    one_hot = F.one_hot(a.view(-1), num_classes=25)
                    data_values = (q_out * one_hot).sum(dim=1, keepdim=True)
                    cql_loss = (logsumexp - data_values).mean()
                    loss = F.mse_loss(q_a, target) + cql_loss
                elif args.scene == "bcq":
                    target = r + args.gamma * max_q_prime * done_mask
                    i_loss = F.nll_loss(imt, a.reshape(-1)).to(dtype=torch.float64) + 1e-2 * i.pow(2).mean()
                    loss = F.mse_loss(q_a, target) + i_loss


                loss_sum += loss.to('cpu').detach()
                value_sum += torch.mean(target).to('cpu').detach()
                rein_q_sum += torch.mean(rein_q).to('cpu').detach()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        local_model.load_state_dict(global_model.state_dict())

        #validataion
        physician_survival_rate, AI_ivf_survival_rate, AI_vaso_survival_rate = validate(global_model, 99, memory, sequence, args)

        #sample tracing
        if epoch == 0 or epoch == 4 or epoch == 9 or epoch == 19 or epoch == 49 or epoch % 100 == 99:
            sample_trace(global_model, 99, memory, sequence, args, log_save_path, epoch)
            sample_trace(global_model, 100, memory, sequence, args, log_save_path, epoch)

        loss_list.append(loss_sum / train_length)
        physician_survival_rate_list.append(physician_survival_rate)
        AI_ivf_survival_rate_list.append(AI_ivf_survival_rate)
        AI_vaso_survival_rate_list.append(AI_vaso_survival_rate)
        rein_q_list.append(rein_q_sum / train_length)
        value_list.append(value_sum / train_length)
        x.append(epoch + 1)



    plt.figure(1)
    my_palette = plt.cm.get_cmap("Set2", 3)
    color = my_palette(0)
    plt.plot(x, physician_survival_rate_list, color=color, alpha=0.8)
    color = my_palette(1)
    plt.plot(x, AI_ivf_survival_rate_list, color=color, alpha=0.8)
    color = my_palette(2)
    plt.plot(x, AI_vaso_survival_rate_list, color=color, alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('survival_rate')
    plt.title('survival_rate')
    plt.legend(['physician_survival_rate', 'AI_ivf_survival_rate', 'AI_vaso_survival_rate'])
    plt.draw()
    fig = plt.gcf()

    fig.savefig(log_save_path + '{} survival rate graph.png'.format(sequence), dpi=300)

    with open(log_save_path + '{}_AI_ivf_survival_rate_list.pickle'.format(sequence), 'wb') as f:
        pickle.dump(AI_ivf_survival_rate_list, f)

    with open(log_save_path + '{}_AI_vaso_survival_rate_list.pickle'.format(sequence), 'wb') as f:
        pickle.dump(AI_vaso_survival_rate_list, f)

    with open(log_save_path + '{}_physician_survival_rate_list.pickle'.format(sequence), 'wb') as f:
        pickle.dump(physician_survival_rate_list, f)
        # print("finished training epoch. {}% done: ".format(epoch / args.max_epoch))

    torch.save(global_model.state_dict(), log_save_path + 'model_{}.pt'.format(sequence))

    return (AI_ivf_survival_rate + AI_vaso_survival_rate) / 2


def validate(global_model, rank, memory, sequence, args):
    local_model = Qnet().to(device)
    local_model.load_state_dict(global_model.state_dict())

    test_length = (memory.val_size() // args.batch_size)
    deeprl2_actions1 = []
    ori_actions1 = []
    ids1 = []
    given_ivs1 = []
    given_vasos1 = []
    mortality_in_test1 = []

    ## To save test data
    ori_action_in_test = []
    action_in_test = []
    mortality_in_test = []
    iv_in_test = []
    vaso_in_test = []
    id_in_test = []
    frame_count_in_test = []

    for epi in range(test_length):
        s, a, r, s_prime, done_mask, death, iv, vaso, id, sq = memory.sample(epi, args.batch_size, rank)

        if args.scene == "bcq":
            q_out, imt, i = global_model.forward(s)
            imt = imt.exp()  # torch.Size([32, 6])
            imt = (imt / imt.max(1, keepdim=True)[0] > args.th).float()
            rein_a = (imt * q_out + (1 - imt) * -1e8).argmax(1, keepdim=True)
            rein_a = torch.reshape(rein_a, [args.batch_size, 1])
            rein_q = q_out.gather(1, rein_a)
        else:
            q_out = global_model(s)
            q_a = q_out.gather(1, a)
            rein_a = torch.argmax(global_model.forward(s), dim=1)
            rein_a = torch.reshape(rein_a, [args.batch_size, 1])
            rein_q = q_out.gather(1, rein_a)


        for i in range(args.batch_size):
            ids1.append(id[i].to('cpu').detach())
            ori_actions1.append(a[i].to('cpu').detach())
            given_ivs1.append(iv[i].to('cpu').detach())
            given_vasos1.append(vaso[i].to('cpu').detach())
            deeprl2_actions1.append(rein_a[i].to('cpu').detach())
            mortality_in_test1.append(death[i].to('cpu').detach())

        ori_action_in_test.append(a.to('cpu').detach())
        action_in_test.append(rein_a.to('cpu').detach())
        mortality_in_test.append(death.to('cpu').detach())
        iv_in_test.append(iv.to('cpu').detach())
        vaso_in_test.append(vaso.to('cpu').detach())
        id_in_test.append(id.to('cpu').detach())

    ids2 = list(itertools.chain.from_iterable(ids1))
    ids2_pd = pd.DataFrame(ids2)
    ids2_pd.columns = ['id']

    ori_actions2 = list(itertools.chain.from_iterable(ori_actions1))
    ori_actions2_pd = pd.DataFrame(ori_actions2)
    ori_actions2_pd.columns = ['ori_actions']

    deeprl2_actions2 = list(itertools.chain.from_iterable(deeprl2_actions1))
    deeprl2_actions2_pd = pd.DataFrame(deeprl2_actions2)
    deeprl2_actions2_pd.columns = ['deeprl2_actions']

    mortality_in_test2 = list(itertools.chain.from_iterable(mortality_in_test1))
    mortality_in_test2_pd = pd.DataFrame(mortality_in_test2)
    mortality_in_test2_pd.columns = ['death']

    ori_actions_tuple = [None for i in range(len(ori_actions2))]
    for i in range(len(ori_actions2)):
        ori_actions_tuple[i] = dict_med[int(ori_actions2[i])]
    ori_actions_tuple_pd = pd.DataFrame(ori_actions_tuple)
    ori_actions_tuple_pd.columns = ['given_vaso', 'given_fluid']

    deeprl2_actions_tuple = [None for i in range(len(deeprl2_actions2))]
    for i in range(len(deeprl2_actions2)):
        deeprl2_actions_tuple[i] = dict_med[int(deeprl2_actions2[i])]
    deeprl2_actions_tuple_pd = pd.DataFrame(deeprl2_actions_tuple)
    deeprl2_actions_tuple_pd.columns = ['model_dose_vaso', 'model_dose_iv']

    vaso = pd.concat([ori_actions_tuple_pd['given_vaso'], deeprl2_actions_tuple_pd['model_dose_vaso']], axis=1)
    vaso['vaso_diff'] = vaso['given_vaso'] - vaso['model_dose_vaso']

    ivf = pd.concat([ori_actions_tuple_pd['given_fluid'], deeprl2_actions_tuple_pd['model_dose_iv']], axis=1)
    ivf['ivf_diff'] = ivf['given_fluid'] - ivf['model_dose_iv']

    optimal_graph = pd.concat([ids2_pd['id'], mortality_in_test2_pd['death'], vaso['vaso_diff'], ivf['ivf_diff']],
                              axis=1)
    val_data = optimal_graph.groupby(['id'], as_index = False).agg({'death':'mean', 'vaso_diff':'mean', 'ivf_diff':'mean'} )
    t = list(range(-1250, 1251, 100))
    t2 = list(range(-105, 106, 10))
    t2[:] = [x / 100 for x in t2]

    physician_mortal = val_data['death'].mean()  # if there are no samples, put average mortality

    AI_ivf = val_data.loc[(val_data['ivf_diff'] > -50) & (val_data['ivf_diff'] < 50)]
    AI_ivf_mortal = AI_ivf['death'].mean()

    AI_vaso = val_data.loc[(val_data['vaso_diff'] > -0.05) & (val_data['vaso_diff'] < 0.05)]
    AI_vaso_mortal = AI_vaso['death'].mean()

    ### physician action in test save #####
    with open(log_save_path + 'ori_action_test_{}.pickle'.format(sequence), 'wb') as f:
        pickle.dump(ori_action_in_test, f)

    ### AI action in test save #####
    with open(log_save_path + 'action_test_{}.pickle'.format(sequence), 'wb') as f:
        pickle.dump(action_in_test, f)

    ### mortality save #####
    with open(log_save_path + 'mortality_test_{}.pickle'.format(sequence), 'wb') as f:
        pickle.dump(mortality_in_test, f)

    ### iv save #####
    with open(log_save_path + 'iv_test_{}.pickle'.format(sequence), 'wb') as f:
        pickle.dump(iv_in_test, f)

    ### vaso save #####
    with open(log_save_path + 'vaso_test_{}.pickle'.format(sequence), 'wb') as f:
        pickle.dump(vaso_in_test, f)

    ### id save #####
    with open(log_save_path + 'id_test_{}.pickle'.format(sequence), 'wb') as f:
        pickle.dump(id_in_test, f)

    with open(log_save_path + 'frame_count_test_{}.pickle'.format(sequence), 'wb') as f:
        pickle.dump(frame_count_in_test, f)

    return (1 - physician_mortal), (1 - AI_ivf_mortal), (1 - AI_vaso_mortal)


def test(global_model, rank, memory, sequence, args, log_save_path):
    local_model = Qnet().to(device)
    local_model.load_state_dict(global_model.state_dict())

    test_length = (memory.test_size() // args.batch_size)

    for epoch in tqdm(range(1), desc='Testing'):
        local_model.load_state_dict(global_model.state_dict())

        ori_action_in_test = []
        action_in_test = []
        mortality_in_test = []
        iv_in_test = []
        vaso_in_test = []
        id_in_test = []
        frame_count_in_test = []

        for epi in range(test_length):
            s, a, r, s_prime, done_mask, death, iv, vaso, id, sq = memory.sample(epi, args.batch_size, rank)

            rein_a = torch.argmax(local_model.forward(s), dim=1)
            rein_a = torch.reshape(rein_a, [-1, 1])

            ori_action_in_test.append(a.to('cpu').detach())
            action_in_test.append(rein_a.to('cpu').detach())
            mortality_in_test.append(death.to('cpu').detach())
            iv_in_test.append(iv.to('cpu').detach())
            vaso_in_test.append(vaso.to('cpu').detach())
            id_in_test.append(id.to('cpu').detach())

    ### physician action in test save #####
    with open(log_save_path + 'ori_action_test_{}.pickle'.format(sequence), 'wb') as f:
        pickle.dump(ori_action_in_test, f)

    ### AI action in test save #####
    with open(log_save_path + 'action_test_{}.pickle'.format(sequence), 'wb') as f:
        pickle.dump(action_in_test, f)

    ### mortality save #####
    with open(log_save_path + 'mortality_test_{}.pickle'.format(sequence), 'wb') as f:
        pickle.dump(mortality_in_test, f)

    ### iv save #####
    with open(log_save_path + 'iv_test_{}.pickle'.format(sequence), 'wb') as f:
        pickle.dump(iv_in_test, f)

    ### vaso save #####
    with open(log_save_path + 'vaso_test_{}.pickle'.format(sequence), 'wb') as f:
        pickle.dump(vaso_in_test, f)

    ### id save #####
    with open(log_save_path + 'id_test_{}.pickle'.format(sequence), 'wb') as f:
        pickle.dump(id_in_test, f)

    with open(log_save_path + 'frame_count_test_{}.pickle'.format(sequence), 'wb') as f:
        pickle.dump(frame_count_in_test, f)


def sample_trace (global_model, rank, memory, sequence, args, log_save_path, epoch):
    local_model = Qnet().to(device)
    local_model.load_state_dict(global_model.state_dict())
    if rank == 100:
        tracing_length = (memory.size() // args.batch_size)
    elif rank == 99:
        tracing_length = (memory.val_size() // args.batch_size)

    trace_lst = []

    for epi in range(tracing_length):
        s, a, r, s_prime, done_mask, death, iv, vaso, id, sq = memory.sample(epi, args.batch_size, rank)

        if args.scene == "bcq":
            q_out, imt, i = global_model.forward(s)
            q_a = q_out.gather(1, a)
            imt = imt.exp()  # torch.Size([32, 6])
            imt = (imt / imt.max(1, keepdim=True)[0] > args.th).float()
            rein_a = (imt * q_out + (1 - imt) * -1e8).argmax(1, keepdim=True)
            rein_a = torch.reshape(rein_a, [args.batch_size, 1])
            rein_q = q_out.gather(1, rein_a)
            trace = torch.cat([id, sq], dim=1)
            trace = torch.cat([trace, rein_q], dim=1)
            trace = torch.cat([trace, q_a], dim=1)
            trace = torch.cat([trace, death], dim=1)
        else:
            q_out = global_model.forward(s)
            q_a = q_out.gather(1, a)
            rein_q, _ = torch.max(q_out, dim=1, keepdim=True)
            trace = torch.cat([id, sq], dim=1)
            trace = torch.cat([trace, rein_q], dim=1)
            trace = torch.cat([trace, q_a], dim=1)
            trace = torch.cat([trace, death], dim=1)

        trace_lst.append(trace.to('cpu').detach())

    if rank == 100:
        with open(log_save_path + '{}_{}_sample_trace_train_list.pickle'.format(sequence, epoch+1), 'wb') as f:
            pickle.dump(trace_lst, f)
    elif rank == 99:
        with open(log_save_path + '{}_{}_sample_trace_test_list.pickle'.format(sequence, epoch+1), 'wb') as f:
            pickle.dump(trace_lst, f)


def training_step (sequence, args, log_save_path):


    global_model = Qnet().to(device)
    global_model.share_memory()  # Global model 공유
    memory = ReplayBuffer(device=device, n_rank=n_rank)

    # data load
    SASR = load_mat_files(file_name=data_path, key_name=data_name)
    SASR = data_norm(SASR, args)
    episodes = split_episodes(SASR)

    # data split and put buffer
    train_idxes, valid_idxes, test_idxes \
        = split_idxes(split_method, n_rank, episodes, split_size, seed=SPLIT_SEED, shuffle=True)

    for rank in range(n_rank):
        for idx in train_idxes:
            for frame in episodes[idx]:
                memory.dqn_train_put(get_data_from_frame(frame))
    for idx in valid_idxes:
        for frame in episodes[idx]:
            memory.dqn_val_put(get_data_from_frame(frame))
    for idx in test_idxes:
        for frame in episodes[idx]:
            memory.dqn_test_put(get_data_from_frame(frame))

    train_length = memory.size()//args.batch_size

    survival_score = train(global_model, memory, train_length, sequence, args, log_save_path)
    # test(global_model, 0, memory, sequence, args, log_save_path)
    return survival_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("--iter", type=int, default=20)
    parser.add_argument("--max-epoch", type=int, default=1) #200
    parser.add_argument("--lr", type=float, default=0.00001)
    parser.add_argument("--clip", type=float, default=3.3) #3.3, 4.0
    parser.add_argument("--isover", type=bool, default=False)
    parser.add_argument("--search", type=bool, default=False)
    parser.add_argument("--highlight-coefficient", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--log-save-dir", type=str, default=f"./result/")
    parser.add_argument("--scene", type=str, default="highlight") #highlight, highreward, cql, bcq
    parser.add_argument("--lr-decay", type=float, default=0.99)
    parser.add_argument("--th", type=float, default=0.3)
    args = parser.parse_args()
    mp.set_start_method('spawn')
    start = time.time()

    log_save_path = os.path.join(
        args.log_save_dir,
        f"scene+{args.scene}highlight_{args.highlight_coefficient}_lr{args.lr}_lr-decay{args.lr_decay}_clip{args.clip}_{str(int(time.time()))[4:]}/"
    )

    makedirs(log_save_path)

    if args.scene == 'bcq':
        from src.Qnet_BCQ import Qnet
    else:
        from src.Qnet import Qnet

    tm = time.ctime(time.time() + 32400)
    print("\nworker start {}, c{} with clip {} at {} .".format(args.scene, args.highlight_coefficient, args.clip, tm))
    count = 1
    train_results = dict()

    while count <= args.iter:
        survival_score = training_step(count, args, log_save_path)
        if args.isover == True:
            print("Overestimation has been detected. End the ongoing training. Trying highlight coefficient {}".format(args.highlight_coefficient))
            count = 1
            args.isover = False
            train_results = dict()
        else:
            print("{} coefficient {} training sequence {} is done in {}.".format(args.scene, args.highlight_coefficient, count, (time.time() - start) / 3600))
            train_results[count] = survival_score
            results_list = sorted(train_results.items(), key = lambda item: item[1], reverse=True)
            df_train_result = pd.DataFrame(results_list)
            df_train_result.to_csv(log_save_path+"0.train_results.csv", header=False, index=False)
            count += 1

