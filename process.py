import logging
import math
import operator
import time
import gol
import torch as t
from util import AverageMeter
import quan

__all__ = ['train', 'validate', 'PerformanceScoreboard']

logger = logging.getLogger()


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with t.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)  # view可能报错，因为用多卡训练的时候tensor不连续，即tensor分布在不同的内存或显存中。
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def init_LSQ_a(model, modules_to_replace, names_to_replace, loader, criterion, monitors, args):
    # 跑一下模型，得出全精度ativation结果
    model.to(args.device.type)
    validate(loader, model, criterion, -1, monitors, args)
    # 得出hook总数
    temp_hook_sum = len(gol.get_value('global_pre_hook_counter'))
    # 得出act总数
    temp_act_sum = len(gol.get_value('global_activation_counter'))
    # 激活结果求均值
    batches_num = temp_act_sum // temp_hook_sum  # 有可能是0
    for iter in range(temp_hook_sum):
        gol.get_value('global_activation')[iter][0] /= batches_num
        gol.get_value('global_activation')[iter][1] /= batches_num
        modules_to_replace[names_to_replace[iter]].quan_a_fn.a_init_from(gol.get_value('global_activation')[iter])


def init_LSQP_a(train_loader, model, args, modules_to_monitor, names_to_monitor):

    optimizer = t.optim.SGD(model.parameters(),
                            lr=args.pre_learning_rate,  # 1 还行，要不2？
                            momentum=args.optimizer.momentum)
    for p in model.parameters():  # 全部置零,有效！
        p.requires_grad = False
        # 遍历来开启grad
    for iter in names_to_monitor:
        modules_to_monitor[iter].quan_a_fn.open_grad()  # 开启scale的grad

    model.to(args.device.type)
    model.train()
    # 进入训练
    for epoch in range(1):
        losses = AverageMeter()
        batch_time = AverageMeter()

        total_sample = len(train_loader.sampler)
        batch_size = train_loader.batch_size
        # steps_per_epoch = math.ceil(total_sample / batch_size)
        logger.info('Tuning quant hypa: %d samples (%d per mini-batch)', total_sample, batch_size)
        end_time = time.time()

        # 生成loss
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(args.device.type)  # 有意思的是，这个是移植到主GPU中的
            outputs = model(inputs)  # 喂数据
            loss = t.tensor(0.0, requires_grad=True)
            # if batch_idx == 0:
            #     print('init loss:', loss)
            for iter in names_to_monitor:
                # print(iter, 'test', loss)
                # print('check loss module:', type(modules_to_monitor[iter]))
                loss = loss + modules_to_monitor[iter].quan_a_fn.norm_loss
                # if batch_idx == 0:
                #     print('get loss:', modules_to_monitor[iter].quan_a_fn.norm_loss)

            losses.update(loss.item(), inputs.size(0))
            optimizer.zero_grad()
            loss.backward()  # 这一步才开始计算梯度
            optimizer.step()
            batch_time.update(time.time() - end_time)
            end_time = time.time()
            if args.dataloader.dataset == 'cifar10':
                if isinstance(model.linear.quan_a_fn, quan.IdentityQuan):
                    logger.info('==> Tuning for hyparm of linear quantizer. epoch:%d batch_id: %d', epoch, batch_idx + 1)
                else:
                    logger.info('==> Tuning for hyparm of linear quantizer. epoch:%d batch_id: %d Loss: %.3f fc scale: %.3f beta: %.6f\n',
                                epoch, batch_idx + 1, losses.avg, model.linear.quan_a_fn.s, model.linear.quan_a_fn.beta)
            elif args.dataloader.dataset == 'imagenet':
                if isinstance(model.fc.quan_a_fn, quan.IdentityQuan):
                    logger.info('==> Tuning for hyparm of linear quantizer. epoch:%d batch_id: %d', epoch, batch_idx + 1)
                else:
                    logger.info('==> Tuning for hyparm of linear quantizer. epoch:%d batch_id: %d Loss: %.3f fc scale: %.3f beta: %.6f\n',
                                epoch, batch_idx + 1, losses.avg, model.fc.quan_a_fn.s, model.fc.quan_a_fn.beta)
            else:
                logger.info(
                    '==> Tuning for hyparm of linear quantizer. epoch:%d batch_id: %d  Loss: %.3f\n',
                    epoch, batch_idx + 1, losses.avg)
            if batch_idx == 500:
                break



def train(train_loader, model, criterion, optimizer, lr_scheduler, epoch, monitors, args):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    batch_time = AverageMeter()

    total_sample = len(train_loader.sampler)
    batch_size = train_loader.batch_size
    steps_per_epoch = math.ceil(total_sample / batch_size)
    logger.info('Training: %d samples (%d per mini-batch)', total_sample, batch_size)

    model.train()
    end_time = time.time()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(args.device.type)
        targets = targets.to(args.device.type)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        acc1, acc5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1.item(), inputs.size(0))
        top5.update(acc5.item(), inputs.size(0))

        if lr_scheduler is not None:
            lr_scheduler.step(epoch=epoch, batch=batch_idx)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        if (batch_idx + 1) % args.log.print_freq == 0:
            for m in monitors:
                m.update(epoch, batch_idx + 1, steps_per_epoch, 'Training', {
                    'Loss': losses,
                    'Top1': top1,
                    'Top5': top5,
                    'BatchTime': batch_time,
                    'LR': optimizer.param_groups[0]['lr']
                })

    logger.info('==> Top1: %.3f    Top5: %.3f    Loss: %.3f\n',
                top1.avg, top5.avg, losses.avg)
    return top1.avg, top5.avg, losses.avg


def validate(data_loader, model, criterion, epoch, monitors, args):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    batch_time = AverageMeter()

    total_sample = len(data_loader.sampler)
    batch_size = data_loader.batch_size
    steps_per_epoch = math.ceil(total_sample / batch_size)

    logger.info('Validation: %d samples (%d per mini-batch)', total_sample, batch_size)
    model.to(args.device.type)
    model.eval()
    end_time = time.time()
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        with t.no_grad():
            inputs = inputs.to(args.device.type)
            targets = targets.to(args.device.type)

            outputs = model(inputs)  # 进入模型推理
            loss = criterion(outputs, targets)

            acc1, acc5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1.item(), inputs.size(0))
            top5.update(acc5.item(), inputs.size(0))
            batch_time.update(time.time() - end_time)
            end_time = time.time()

            if (batch_idx + 1) % args.log.print_freq == 0:
                for m in monitors:
                    m.update(epoch, batch_idx + 1, steps_per_epoch, 'Validation', {
                        'Loss': losses,
                        'Top1': top1,
                        'Top5': top5,
                        'BatchTime': batch_time
                    })

    logger.info('==> Top1: %.3f    Top5: %.3f    Loss: %.3f\n', top1.avg, top5.avg, losses.avg)
    return top1.avg, top5.avg, losses.avg


class PerformanceScoreboard:
    def __init__(self, num_best_scores):
        self.board = list()
        self.num_best_scores = num_best_scores

    def update(self, top1, top5, epoch):
        """ Update the list of top training scores achieved so far, and log the best scores so far"""
        self.board.append({'top1': top1, 'top5': top5, 'epoch': epoch})

        # Keep scoreboard sorted from best to worst, and sort by top1, top5 and epoch
        curr_len = min(self.num_best_scores, len(self.board))
        self.board = sorted(self.board,
                            key=operator.itemgetter('top1', 'top5', 'epoch'),
                            reverse=True)[0:curr_len]
        for idx in range(curr_len):
            score = self.board[idx]
            logger.info('Scoreboard best %d ==> Epoch [%d][Top1: %.3f   Top5: %.3f]',
                        idx + 1, score['epoch'], score['top1'], score['top5'])

    def is_best(self, epoch):
        return self.board[0]['epoch'] == epoch
