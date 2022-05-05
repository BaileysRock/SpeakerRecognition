import time
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter


def train(config, model, train_iter, dev_iter):
    start_time = time.time()
    # 启用dropout
    model.train()
    # 设置adam优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    total_batch = 0                # 记录总共训练的批次
    dev_best_loss = float('inf')   # 记录验证集上最低的loss
    dev_best_acc = float(0)        # 记录验证集上最高的acc
    last_improve = 0               # 记录上一次dev的loss下降时的批次
    flag = False                   # 是否结束训练
    writer = SummaryWriter(log_dir=config.log_path + time.strftime('%m-%d_%H.%M', time.localtime()))
    for epoch in range(config.epoch):
        print("Epoch [{}/{}]".format(epoch+1, config.epoch))
        for i, trains in enumerate(train_iter):
            trains['wav1'] = trains['wav1'].to(config.device)
            trains['wav2'] = trains['wav2'].to(config.device)
            trains['label'] = trains['label'].to(config.device)
            outputs = model(trains)
            # print(outputs)
            model.zero_grad()
            loss = F.mse_loss(outputs[0], outputs[1])
            lossList = []
            len = outputs[0].shape[0]
            for j in range(len):
                for k in range(len):
                    if k != j:
                        lossList.append(F.triplet_margin_loss(outputs[j][0], outputs[j][1], outputs[k][0]))
                        lossList.append(F.triplet_margin_loss(outputs[j][0], outputs[j][1], outputs[k][1]))
            for lossItem in lossList:
                loss += lossItem
            loss.backward()
            optimizer.step()
            # 输出当前效果
            if total_batch % 10 == 0:
                dev_acc, dev_loss = evaluate(model,config, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                if dev_acc > dev_best_acc:
                    dev_best_acc = dev_acc
                    torch.save(model.state_dict(), config.save_model_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                print("Iter:{:4d} TrainLoss:{:.12f} DevLoss:{:.12f} DevAcc:{:.5f} Improve:{}".format(total_batch, loss.item(), dev_loss, dev_acc * 100, improve))
                writer.add_scalar("loss/train", loss.item(), total_batch)
                writer.add_scalar("loss/dev", dev_loss, total_batch)
                writer.add_scalar("acc/dev", dev_acc, total_batch)
                model.train()
            total_batch += 1
            if total_batch - last_improve > config.maxiter_without_improvement:
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
    writer.close()
    end_time = time.time()
    print("Train Time : {:.3f} min , The Best Acc in Dev : {} % , The Best Loss in Dev : {} % ".format(((float)((end_time-start_time))/60), dev_best_acc * 100, dev_best_loss))



def evaluate(model,config, evalDataLoader):
    wav1List = []
    wav2List = []
    loss = 0
    for i, evals in enumerate(evalDataLoader):
        evals['wav1'] = evals['wav1'].to(config.device)
        evals['wav2'] = evals['wav2'].to(config.device)
        evals['label'] = evals['label'].to(config.device)
        outputs = model(evals)
        outputs[0].cpu()
        outputs[1].cpu()
        loss += F.l1_loss(outputs[0], outputs[1], reduction='sum').item()
        wav1List.extend(data.cpu() for data in outputs[0])
        wav2List.extend(data.cpu() for data in outputs[1])
    length = len(wav1List)
    matched = 0
    for i in range(length):
        wav1 = wav1List[i]
        match = i
        matchLoss = float('inf')
        for j in range(length):
            if F.l1_loss(wav1,wav2List[j]).item() < matchLoss:
                matchLoss = F.l1_loss(wav1,wav2List[j]).item()
                match = j
        if i == match:
            matched += 1
    acc = (float)(matched) / (float)(length)
    return acc, loss



