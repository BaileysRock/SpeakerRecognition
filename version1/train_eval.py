import time
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter


def train(config,model,train_iter,dev_iter):
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
            outputs = model(trains)
            model.zero_grad()
            loss = F.l1_loss(outputs[0],outputs[1],reduction='sum')
            loss.backward()
            optimizer.step()
            # 输出当前效果
            if total_batch % 10 == 0:
                dev_acc, dev_loss = evaluate(model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                if dev_acc > dev_best_acc:
                    dev_best_acc = dev_acc
                    torch.save(model.state_dict(), config.save_model_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                print("Iter:{:4d} TrainLoss:{:.12f} DevLoss:{:.12f} DevAcc:{:.5f} Improve:{}".format(total_batch,loss.item(),dev_loss,dev_acc,improve))
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
    print("Train Time : {:.3f} min , The Best Acc in Dev : {} % , The Best Loss in Dev : {} % ".format(((float)((end_time-start_time))/60), dev_best_acc,dev_best_loss))






def evaluate(model, evalDataLoader):
    wav1List = []
    wav2List = []
    loss = 0
    for i, evals in enumerate(evalDataLoader):
        outputs = model(evals)
        loss += F.l1_loss(outputs[0], outputs[1], reduction='sum').item()
        wav1List.extend(data for data in outputs[0])
        wav2List.extend(data for data in outputs[1])
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



