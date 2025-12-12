import sys
import time
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from GLADS_data_handler import pcap2npy_GLADS, json2npy_GLADS, KFoldIdxGen, npy2dataloader
from thop import profile
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch.optim.lr_scheduler import MultiStepLR
import warnings

warnings.filterwarnings('ignore')

class GL_multiAttention(nn.Module):
    def __init__(self, batch_size, unit_num, filters, head_num):
        super(GL_multiAttention, self).__init__()
        self.filters = filters
        self.head_num = head_num
        self.Wq = nn.Linear(filters, filters)
        self.Wv = nn.Linear(filters, filters)
        self.div = torch.arange(1, unit_num + 1).view(-1, 1).expand(-1, filters).to(device)
        self.M = (torch.triu(torch.ones(head_num, unit_num, unit_num), diagonal=1) * minf).to(device)

    def forward(self, x): #input: [B, c, n]
        K = torch.transpose(x, -1, -2) #[B, n, c]
        G = torch.cumsum(K, dim=1) / self.div #[B, n, c]
        Q = self.Wq(G)
        V = self.Wv(K)
        H_size = self.filters // self.head_num
        querys = torch.stack(torch.split(Q, H_size, dim=2), dim=1) #[B, head_num, n, H_size]
        keys = torch.stack(torch.split(K, H_size, dim=2), dim=1) #[B, head_num, n, H_size]
        values = torch.stack(torch.split(V, H_size, dim=2), dim=1) #[B, head_num, n, H_size]

        scores = torch.matmul(querys, torch.transpose(keys, -1, -2)) #[B, head_num, n, n]
        scores[scores <= thresd] = minf #公式(15)
        scores = scores + self.M #加上公式(20)
        scores = nn.functional.softmax(scores / (H_size ** 0.5), dim=-1)
        out = torch.matmul(scores, values) #公式(22) [B, head_num, n, H_size]
        out = torch.cat(torch.split(out, 1, dim=1), dim=3).squeeze(dim=1) #[B,n,c]
        return out

class Base_Block(nn.Module):
    def __init__(self, filters):
        super(Base_Block, self).__init__()
        self.DepthwiseCNN = nn.Sequential(
            nn.Conv1d(in_channels=filters, out_channels=filters, groups=filters, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.BatchNorm1d(filters)
        )
        self.PointwiseCNN = nn.Sequential(
            nn.Conv1d(in_channels=filters, out_channels=filters, groups=1, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            nn.BatchNorm1d(filters)
        )
        self.final_activation = nn.GELU()
    def forward(self, x):
        x = x + self.DepthwiseCNN(x)
        x = x + self.PointwiseCNN(x)
        return self.final_activation(x)

class GLADS(nn.Module):
    def __init__(self, batch_size, input_len, filters):
        super(GLADS, self).__init__()
        self.embeding = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=filters, kernel_size=4, stride=2, padding=0),
            nn.GELU(),
            nn.BatchNorm1d(filters),

        )
        self.exchange_layer = nn.Sequential(
            Base_Block(filters), # 1
            Base_Block(filters), # 2
            Base_Block(filters), # 3
            Base_Block(filters), # 4
            Base_Block(filters), # 5
            Base_Block(filters), # 6
            Base_Block(filters), # 7
        )
        self.merge = nn.Conv1d(filters, filters, kernel_size=2, stride=2, groups=filters)
        self.att = GL_multiAttention(batch_size, (input_len // 2 -1) //2, filters, 16)
        self.fc1 = nn.Sequential(
            nn.Linear(filters, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, tasknums[0])
        )
        # self.fc2 = nn.Sequential(
        #     nn.Linear(filters, 128),
        #     nn.ReLU(),
        #     nn.Dropout(0.2),
        #     nn.Linear(128, tasknums[1])
        # )
        # self.fc3 = nn.Sequential(
        #     nn.Linear(filters, 128),
        #     nn.ReLU(),
        #     nn.Dropout(0.2),
        #     nn.Linear(128, tasknums[2])
        # )
    def forward(self, x):
        x_embed = self.embeding(x)
        x_exg = self.exchange_layer(x_embed)
        x_merge = self.merge(x_exg)
        x_att = self.att(x_merge)
        x_select = x_att[:, -1].squeeze(dim=1)
        out1 = self.fc1(x_select)
        # out2 = self.fc2(x_select)
        # out3 = self.fc3(x_select)
        return out1




def test_model_new(model, x_test, y_test, class_num:list, record, rnd):
    model.eval()
    testLoader = npy2dataloader(x_test, y_test, batch_size=batch_size, shuffle=True)
    task_num = len(class_num)


    now_rcd = np.zeros((task_num, 5), dtype=np.float32)
    total_time = 0
    pnum = 0
    for data, label in testLoader:
        pnum += data.shape[0]
        before_eval = time.time()
        data = data.to(device)
        label = label.to(device)
        t1_pre= model(data)
        after_eval = time.time()
        total_time += after_eval - before_eval
        pres = [t1_pre]
        pres = [torch.argmax(nn.functional.softmax(x, dim=1), dim=1) for x in pres]

        predictions = [x.detach().cpu().numpy() for x in pres]
        truelabels = [label[:, tt].detach().cpu().numpy() for tt in range(task_num)]

        for t in range(task_num):
            cnf_matrix = confusion_matrix(y_true=truelabels[t], y_pred=predictions[t])

            FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
            FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
            TP = np.diag(cnf_matrix)
            TN = cnf_matrix.sum() - (FP + FN + TP)

            FP = FP.astype(float)
            FN = FN.astype(float)
            TP = TP.astype(float)
            TN = TN.astype(float)
            TPR = np.nan_to_num(TP / (TP + FN)) #recall
            FPR = np.nan_to_num(FP / (FP + TN))

            FTF = 0
            weight = cnf_matrix.sum(axis=1)
            w_sum = weight.sum(axis=0)

            for i in range(len(weight)):
                FTF += weight[i] * TPR[i] / (1 + FPR[i])
            FTF /= w_sum
            FPR = float(str(np.around(np.mean(FPR), decimals=4).tolist()))
            TPR = float(str(np.around(np.mean(TPR), decimals=4).tolist()))
            FTF = float(str(np.around(FTF, decimals=4)))
            # print(float(str(np.around(np.mean(FPR), decimals=4).tolist())), float(str(np.around(np.mean(TPR), decimals=4).tolist())), float(str(np.around(FTF, decimals=4))))


            Acc = accuracy_score(y_true=truelabels[t], y_pred=predictions[t])
            Pre = precision_score(y_true=truelabels[t], y_pred=predictions[t],
                                  average=('macro' if class_num[t] > 2 else 'binary'))
            Rec = recall_score(y_true=truelabels[t], y_pred=predictions[t],
                               average=('macro' if class_num[t] > 2 else 'binary'))
            F1 = f1_score(y_true=truelabels[t], y_pred=predictions[t],
                          average=('macro' if class_num[t] > 2 else 'binary'))

            #print(float(str(np.around(np.mean(TPR), decimals=4).tolist())), Rec)

            # return float(str(np.around(np.mean(FPR), decimals=4).tolist())), float(
            #     str(np.around(np.mean(TPR), decimals=4).tolist())), \
            #     float(str(np.around(FTF, decimals=4)))

            rst = np.array([Acc, TPR, FPR, FTF, F1], dtype=np.float32)
            now_rcd[t] = now_rcd[t] + rst

    print('eval time:', total_time / pnum * 1000)

    now_rcd = now_rcd / len(testLoader)
    for t in range(task_num):
        print('accuracy:%.4f' % now_rcd[t][0])
        print('TPR:%.4f' % now_rcd[t][1])
        print('FPR:%.4f' % now_rcd[t][2])
        print('FTF:%.4f' % now_rcd[t][3])
        print('F1:%.4f' % now_rcd[t][4])
        if now_rcd[t][0] > record[rnd, t, 0]:
            record[rnd, t, 0] = now_rcd[t][0]
            record[rnd, t, 1] = now_rcd[t][1]
            record[rnd, t, 2] = now_rcd[t][2]
            record[rnd, t, 3] = now_rcd[t][3]
            record[rnd, t, 4] = now_rcd[t][4]



n_fold = 5
Nb = 784
Np = 32
feature_len = Nb + 8*Np
batch_size = 128
epoches = 60
thresd = 0 #公式(15)的t
minf = 1 - (2 ** 32)
tasknums = [25]

base = r"E:\Acolasian\datasets\Bytedance\self_tor"
strat_idx = 5



print('----------------split dataset----------------------')
gen, sample_all, label_all = KFoldIdxGen(base, start=strat_idx, num=len(tasknums), n_split=n_fold, shuffle=True, sample_num=0, upper=0, lower=0)

print('------------------pcap2npy-------------------------')
sample_npy = pcap2npy_GLADS(sample_all, Np=Np, Nb=Nb) #读取json文件时使用json2npy_GLADS
sample_npy = np.expand_dims(sample_npy, axis=1)
print(sample_npy.shape)

# print(sample_all[0])
# print(sample_npy[0, 0, :5, :50])
# print(sample_npy[0, 1, :5, :50])


print('-------------------build model---------------------')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GLADS(batch_size, feature_len, 96).to(device)
criterion = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
scheduler = MultiStepLR(opt, milestones=[30, 45, 55], gamma=0.3)


test_input = torch.randn(1, 1, feature_len)
test_input = test_input.to(device)
flops, params = profile(model, (test_input,))
print('flops:%.2fM params:%.2fM' % (flops/1e6, params/1e6))


print('--------------training model-----------------------')


best_record = np.zeros((n_fold, len(tasknums), 5), dtype=np.float32)

for rd, (train_idx, test_idx) in enumerate(gen):
    print('-------------Round%d-----------' % (rd+1))

    # for p in model.parameters():
    #     nn.init.kaiming_uniform_(p)
    # opt = torch.optim.AdamW(model.parameters(), lr=0.001)

    #todo:尝试在每轮开始之前初始化模型参数，失败。遂在每轮开始之前重新创建模型
    model = GLADS(batch_size, feature_len, 96).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = MultiStepLR(opt, milestones=[30, 45, 55], gamma=0.3)


    trainLoader = npy2dataloader(sample_npy[train_idx], label_all[train_idx], batch_size=batch_size, shuffle=True)


    for epoch in range(epoches):
        model.train()
        epoch_loss = 0.0
        before_train = time.time()
        spnum = 0
        for data, label in tqdm(trainLoader, total=len(trainLoader)):
            spnum += data.shape[0]
            data = data.to(device)
            label = label.to(device)

            opt.zero_grad()
            out = model(data)
            loss = criterion(out, label[:, 0])
            loss.backward()

            epoch_loss += loss.item()

            opt.step()
        scheduler.step()
        after_train = time.time()
        print('train time:', (after_train - before_train) / spnum * 128 * 100)
        test_model_new(model, sample_npy[test_idx], label_all[test_idx], tasknums, best_record, rd)
        print(f"Epoch {epoch + 1}/{epoches}, Loss: {epoch_loss / len(trainLoader):.4f}")
        print('gpu mem:', torch.cuda.max_memory_allocated())
best_record = best_record * 100
for t in range(1):
    print(f'T{t + 1} Acc: {np.mean(best_record[:, t, 0]):.2f}±{np.std(best_record[:, t, 0]):.2f}')
    print(f'T{t + 1} TPR: {np.mean(best_record[:, t, 1]):.2f}±{np.std(best_record[:, t, 1]):.2f}')
    print(f'T{t + 1} FPR: {np.mean(best_record[:, t, 2]):.2f}±{np.std(best_record[:, t, 2]):.2f}')
    print(f'T{t + 1} FTF: {np.mean(best_record[:, t, 3]):.2f}±{np.std(best_record[:, t, 3]):.2f}')
    print(f'T{t + 1} F1:  {np.mean(best_record[:, t, 4]):.2f}±{np.std(best_record[:, t, 4]):.2f}')