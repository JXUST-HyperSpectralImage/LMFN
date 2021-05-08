# 定义普通训练过程
import os
import joblib
import torch
from torch import optim
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from utils import count_sliding_window, sliding_window, grouper, camel_to_snake


# 普通训练过程
def train(logger, net, optimizer, criterion, train_loader, epoch, save_epoch, scheduler=None,
          device=torch.device('cpu'), val_loader=None, supervision='full', vis_display=None, RUN=None):
    # 首先检查损失函数
    if criterion is None:
        logger.debug("Missing criterion. You must specify a loss function.")
        raise Exception("Missing criterion. You must specify a loss function.")

    # 定义全局变量
    net.to(device)
    save_epoch = save_epoch if epoch > 20 else 1
    lr_list = []
    losses = np.zeros(1000000)
    avg_losses = np.zeros(10000)
    iter_ = 0
    batch_loss_win, epoch_loss_win, val_win, lr_win = None, None, None, None
    val_accuracies = []
    LEN = len(train_loader)

    for e in tqdm(range(1, epoch + 1), desc="Training the network"):
        # 因为每轮训练结尾都要进行验证模式，需要重新将模式调整回训练模式
        net.train()
        avg_loss = 0.
        for batch_idx, (data, label) in enumerate(train_loader):
            # 将数据载入GPU
            data, label = data.to(device), label.to(device)
            optimizer.zero_grad()
            if supervision == 'full':
                output = net(data)
                loss = criterion(output, label)
            elif supervision == 'semi':
                output = net(data)
                output, rec = output
                loss = criterion[0](output, label) + net.aux_loss_weight * criterion[1](rec, data)
            else:
                raise ValueError("supervision mode \"{}\" is unknown.".format(supervision))
            loss.backward()
            optimizer.step()
            #scheduler.step()

            avg_loss += loss.item()
            losses[iter_] = loss.item()
            lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])

            iter_ += 1
            # 释放缓存
            del (data, label, loss, output)

        avg_loss /= LEN
        avg_losses[e] = avg_loss
        if val_loader is not None:
            val_acc = val(net, val_loader, device=device, supervision=supervision)
            val_accuracies.append(val_acc)
            # 因为这里的精度要在后面的梯度更新中用到，所以取负数，metric越小说明
            metric = -val_acc
        else:
            metric = avg_loss
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(metric)
        elif scheduler is not None:
            scheduler.step()
        # 在控制台打印信息
        tqdm.write(f"Epoch [{e}/{epoch}    avg_loss:{avg_loss:.2f}, val_acc:{val_acc:.2f}]")
        # 在日志打印信息
        logger.debug(f"Epoch [{e}/{epoch}    avg_loss:{avg_loss:.2f}, val_acc:{val_acc:.2f}]")
        # 保存断点(如果不是必要情况下不用保存，保存会降低模型训练速度)
        #        if e%save_epoch == 0:
        #            update = None if loss_win is None else 'append'
        #            save_model(logger, net, camel_to_snake(str(net.__class__.__name__)),train_loader.dataset.dataset_name, epoch=e, metric=abs(metric))
        if e % save_epoch == 0:
            epoch_loss_win = vis_display.line(
                X=np.arange(e),
                Y=avg_losses[:e],
                win=epoch_loss_win,
                opts={'title': "Epoch loss"+str(RUN),
                      'xlabel': "Iterations",
                      'ylabel': "Loss"
                      })
            val_win = vis_display.line(Y=np.array(val_accuracies),
                                       X=np.arange(len(val_accuracies)),
                                       win=val_win,
                                       opts={'title': "Validation accuracy"+str(RUN),
                                             'xlabel': "Epochs",
                                             'ylabel': "Accuracy"
                                             })
    batch_loss_win = vis_display.line(
        X=np.arange(iter_),
        Y=losses[:iter_],
        win=batch_loss_win,
        opts={'title': "Batch loss"+str(RUN),
              'xlabel': "Iterations",
              'ylabel': "Loss"
              })
    # lr_win = vis_display.line(
    #     X=np.arange(len(lr_list)),
    #     Y=np.array(lr_list),
    #     win=lr_win,
    #     opts={'title': "Learning rate"+str(RUN),
    #           'xlabel': "Iterations",
    #           'ylabel': "LR"
    #           })


# 普通验证过程
def val(net, data_loader, device='cpu', supervision='full'):
    accuracy, total = 0., 0.
    ignored_labels = data_loader.dataset.ignored_labels
    for batch_idx, (data, target) in enumerate(data_loader):
        with torch.no_grad():
            # Load the data into the GPU if required
            data, target = data.to(device), target.to(device)
            if supervision == 'full':
                output = net(data)
            elif supervision == 'semi':
                outs = net(data)
                output, rec = outs
            _, output = torch.max(output, dim=1)
            # target = target - 1
            for out, pred in zip(output.view(-1), target.view(-1)):
                if out.item() in ignored_labels:
                    continue
                else:
                    accuracy += out.item() == pred.item()
                    total += 1
    return accuracy / total


# 测试过程
def test(net, img, hyperparams):
    """
    Test a model on a specific image
    """
    net.eval()
    patch_size = hyperparams['patch_size']
    center_pixel = hyperparams['center_pixel']
    batch_size, device = hyperparams['batch_size'], hyperparams['device']
    n_classes = hyperparams['n_classes']

    kwargs = {'step': hyperparams['test_stride'], 'window_size': (patch_size, patch_size)}
    probs = np.zeros(img.shape[:2] + (n_classes,))
    # 统计生成窗口的数量 // bath_size = 迭代次数
    iterations = count_sliding_window(img, **kwargs) // batch_size
    # grouper生成batch_size数量的窗口
    for batch in tqdm(grouper(batch_size, sliding_window(img, **kwargs)),
                      # ncols=iterations,
                      total=(iterations),
                      desc="Inference on the image"
                      ):
        with torch.no_grad():
            if patch_size == 1:
                data = [b[0][0, 0] for b in batch]  # 如果是一维数据，则每次只取最左上角的数据
                data = np.copy(data)
                data = torch.from_numpy(data)
            else:
                data = [b[0] for b in batch]  # 每次取整个窗口数据
                data = np.copy(data)
                data = data.transpose(0, 3, 1, 2)  # (N, H, W, C)-->（N, C, H, W)
                data = torch.from_numpy(data)
                if hyperparams['patch_size']>1:
                    data = data.unsqueeze(1)

            indices = [b[1:] for b in batch]
            data = data.to(device)
            output = net(data)
            if isinstance(output, tuple):
                output = output[0]
            output = output.to('cpu')

            if patch_size == 1 or center_pixel:
                output = output.numpy()
            else:
                output = np.transpose(output.numpy(), (0, 2, 3, 1))
            for (x, y, w, h), out in zip(indices, output):
                if center_pixel:
                    probs[x + w // 2, y + h // 2] += out
                else:
                    probs[x:x + w, y:y + h] += out
    return probs


# 保存模型
def save_model(logger, model, model_name, dataset_name, **kwargs):
    model_dir = './checkpoints/' + model_name + "/" + dataset_name + "/"
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    if isinstance(model, torch.nn.Module):
        filename = str('run') + "_epoch{epoch}_{metric:.2f}".format(**kwargs)
        tqdm.write("Saving neural network weights in {}".format(filename))
        logger.debug("-----Saving neural network weights in {}-----".format(filename))
        torch.save(model.state_dict(), model_dir + filename + '.pth')
    else:
        filename = str('run')
        tqdm.write("Saving model params in {}".format(filename))
        logger.debug("-----Saving model params in {}-----".format(filename))
        joblib.dump(model, model_dir + filename + '.pkl')
