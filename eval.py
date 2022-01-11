import logging
from pathlib import Path
import os
import torch as t
import yaml

import process
import quan
import util
from model import create_model
import gol

def main():
    gol._init()
    gol.set_value('global_pre_hook', [])
    gol.set_value('global_activation', [])
    gol.set_value('global_pre_hook_counter', [])
    gol.set_value('global_activation_counter', [])

    script_dir = Path.cwd()  # 获取该main.py文件所在的路径
    args = util.get_config(default_file=script_dir / 'config.yaml')

    output_dir = script_dir / args.output_dir  # 神奇，好像是因为path对象才可以这样子更新路径。
    output_dir.mkdir(exist_ok=True)  #　建立文件夹

    log_dir = util.init_logger(args.name, output_dir, script_dir / 'logging.conf')
    logger = logging.getLogger()
    # 下面这个代码其实就是想转存yaml文件而已
    with open(log_dir / "args.yaml", "w") as yaml_file:  # dump experiment config
        yaml.safe_dump(args, yaml_file)

    pymonitor = util.ProgressMonitor(logger)  # 创建一个过程监视器，其实也是绑定log，监视train过程
    tbmonitor = util.TensorBoardMonitor(logger, log_dir)  # 创建一个tb监视器，监视tensorboard输出信息
    monitors = [pymonitor, tbmonitor]  #　包在一起
    # 设置一下device的可见性
    temp = ''
    temp_length = len(args.device.gpu)
    for i in range(len(args.device.gpu)):
        if i is not 0:
            temp += ','
        temp += str(args.device.gpu[i])
    os.environ['CUDA_VISIBLE_DEVICES'] = temp
    args.device.gpu = []
    for i in range(temp_length):
        args.device.gpu.append(i)

    if args.device.type == 'cpu' or not t.cuda.is_available() or args.device.gpu == []:
        args.device.gpu = []
    else:
        available_gpu = t.cuda.device_count()
        for dev_id in args.device.gpu:
            if dev_id >= available_gpu:
                logger.error('GPU device ID {0} requested, but only {1} devices available'
                             .format(dev_id, available_gpu))
                exit(1)
        # Set default device in case the first one on the list
        # t.cuda.set_device(args.device.gpu[0])  # 这是进行绑定，绑定要使用的GPU卡
        # Enable the cudnn built-in auto-tuner to accelerating training, but it
        # will introduce some fluctuations in a narrow range.
        t.backends.cudnn.benchmark = True  # 实现网络的加速。但是做剪枝的时候不要用这个。
        t.backends.cudnn.deterministic = False  # 每次返回的卷积算法将是确定的，即默认算法。这就避免了其他参数相同的时候train出来还不一样

    # Initialize simple data loader
    train_loader, val_loader, test_loader = util.load_data(args.dataloader)  #　加载数据，生成train、val、test。具体是什么数据看arg
    logger.info('simple Dataset `%s` size:' % args.dataloader.dataset +
                '\n          Training Set = %d (%d)' % (len(train_loader.sampler), len(train_loader)) +
                '\n        Validation Set = %d (%d)' % (len(val_loader.sampler), len(val_loader)) +
                '\n              Test Set = %d (%d)' % (len(test_loader.sampler), len(test_loader)))

    # Create the model
    model = create_model(arch=args.arch, dataset=args.dataloader.dataset, pre_trained=args.pre_trained,
                         hub_path=args.hub_path)

    # 创建对应的量化层（其实量化层就是挂载了量化器的层）
    modules_to_replace, ori_modules, names_to_replace = quan.find_modules_to_quantize(model, args.quan)
    # Define criterion
    criterion = t.nn.CrossEntropyLoss().to(args.device.type)

    # 先得出预训练模型的情况
    logger.info('Get Pretrained Model Score')
    # model.to(args.device.type)
    # process.validate(test_loader, model, criterion, -2, monitors, args)  # TODo 变回来!
    # 因为act的初始化问题而有
    # act_mode = args.quan.act.mode
    # if act_mode == 'lsq':
    #     # 用于以数据初始化
    #     quan.hang_the_hook(ori_modules, names_to_replace, quan.get_fm_hook)
    #     # 对act量化器进行初始化
    #     process.init_LSQ_a(
    #         model, modules_to_replace, names_to_replace, val_loader, criterion, monitors, args)
    #     # 移除所有的钩子
    #     del gol.get_value('global_pre_hook')[:]  # 变成[]
    #     del gol.get_value('global_pre_hook_counter')[:]
    #     # 将量化层替换到模型中
    #     model = quan.replace_module_by_names(model, modules_to_replace)
    #
    # elif act_mode == 'lsqp':
    #     # 将量化层替换到模型中
    #     model = quan.replace_module_by_names(model, modules_to_replace)
    #     # 用于在refine下初始化
    #     quan.hang_the_hook(modules_to_replace, names_to_replace, quan.refine_hypa_hook)
    #     process.init_LSQP_a(
    #         train_loader, model, args, modules_to_replace, names_to_replace)
    #     for p in model.parameters():  # 全部启动,有效！
    #         p.requires_grad = True
    #     # 移除所有的钩子
    #     del gol.get_value('global_pre_hook')[:]  # 变成[]
    #     del gol.get_value('global_pre_hook_counter')[:]
    #
    # else:
    #     raise ValueError('Cannot find quan mode `%s`', act_mode)

    # 这个add_graph操作就是传进去一点数据，然后打通计算图。其实这个是拿来检验量化层是否成功替换的好方法。使用时必须保证在同一device上！
    # model.to('cpu')
    # model.eval()
    # tbmonitor.writer.add_graph(model, input_to_model=train_loader.dataset[0][0].unsqueeze(0))
    # logger.info('Inserted quantizers into the original model')

    model = quan.replace_module_by_names(model, modules_to_replace)
    # epoch计数器
    # start_epoch = 0
    # 是否继续上一次QAT训练（或导入完成量化的模型）
    if args.resume.path:
        model, start_epoch, _ = util.load_checkpoint(
            model, args.resume.path, args.device.type, lean=args.resume.lean, strict=False)
    # 如果有多个卡，就并行分布
    if args.device.gpu and not args.dataloader.serialized:
        model = t.nn.DataParallel(model, device_ids=args.device.gpu)  # 这会套一个外壳，parallel类
    model.to(args.device.type)



    # Initialize final data loader
    del test_loader, val_loader, train_loader
    args.dataloader.batch_size = args.final_batch_size
    train_loader, val_loader, test_loader = util.load_data(args.dataloader)  # 加载数据，生成train、val、test。具体是什么数据看arg
    logger.info('Final Dataset `%s` size:' % args.dataloader.dataset +
                '\n          Training Set = %d (%d)' % (len(train_loader.sampler), len(train_loader)) +
                '\n        Validation Set = %d (%d)' % (len(val_loader.sampler), len(val_loader)) +
                '\n              Test Set = %d (%d)' % (len(test_loader.sampler), len(test_loader)))

    # 定义优化器。注意了，由于需要绑定模型的parameter，所以注意使用顺序！！！！
    optimizer = t.optim.SGD(model.parameters(),
                            lr=args.optimizer.learning_rate,
                            momentum=args.optimizer.momentum,
                            weight_decay=args.optimizer.weight_decay)  # 注意，这里给的weight_decay要合理！
    lr_scheduler = util.lr_scheduler(optimizer,
                                     batch_size=train_loader.batch_size,
                                     num_samples=len(train_loader.sampler),
                                     **args.lr_scheduler)
    logger.info(('Optimizer: %s' % optimizer).replace('\n', '\n' + ' ' * 11))
    logger.info('LR scheduler: %s\n' % lr_scheduler)
    # perf_scoreboard = process.PerformanceScoreboard(args.log.num_best_scores)  # 这是干嘛用的？
    # 运行模式，eval or train
    if args.eval:
        process.validate(test_loader, model, criterion, -1, monitors, args)
    # else:  # training
    #     if args.resume.path or args.pre_trained:  # 如果使用了训练过的模型的话，先eval一下替换量化层后的情况
    #         logger.info('>>>>>>>> Epoch -1 (pre-trained model evaluation)')
    #         top1, top5, _ = process.validate(val_loader, model, criterion,
    #                                          start_epoch - 1, monitors, args)
    #         perf_scoreboard.update(top1, top5, start_epoch - 1)
    #
    #     for epoch in range(start_epoch, args.epochs):
    #         logger.info('>>>>>>>> Epoch %3d' % epoch)
    #         # 监视一下
    #         if args.dataloader.dataset == 'cifar10':
    #             if isinstance(model.module.linear.quan_a_fn, quan.IdentityQuan):
    #                 pass
    #             else:
    #                 try:
    #                     logger.info('>>>>>>>> linear activation scale: %3f  beta: %6f', model.module.linear.quan_a_fn.s,
    #                                 model.module.linear.quan_a_fn.beta)
    #                     logger.info('>>>>>>>> linear weight scale: %3f', model.module.linear.quan_w_fn.s)
    #                 except:
    #                     logger.info('>>>>>>>> linear activation scale: %3f', model.module.linear.quan_a_fn.s)
    #                     logger.info('>>>>>>>> linear weight scale: %3f', model.module.linear.quan_w_fn.s)
    #         elif args.dataloader.dataset == 'imagenet':
    #             if isinstance(model.module.fc.quan_a_fn, quan.IdentityQuan):
    #                 pass
    #             else:
    #                 try:
    #                     logger.info('>>>>>>>> linear activation scale: %3f  beta: %6f', model.module.fc.quan_a_fn.s,
    #                                 model.module.fc.quan_a_fn.beta)
    #                     logger.info('>>>>>>>> linear weight scale: %3f', model.module.fc.quan_w_fn.s)
    #                 except:
    #                     logger.info('>>>>>>>> linear activation scale: %3f', model.module.fc.quan_a_fn.s)
    #                     logger.info('>>>>>>>> linear weight scale: %3f', model.module.fc.quan_w_fn.s)
    #
    #         t_top1, t_top5, t_loss = process.train(train_loader, model, criterion, optimizer,
    #                                                lr_scheduler, epoch, monitors, args)
    #         v_top1, v_top5, v_loss = process.validate(val_loader, model, criterion, epoch, monitors, args)
    #
    #         tbmonitor.writer.add_scalars('Train_vs_Validation/Loss', {'train': t_loss, 'val': v_loss}, epoch)
    #         tbmonitor.writer.add_scalars('Train_vs_Validation/Top1', {'train': t_top1, 'val': v_top1}, epoch)
    #         tbmonitor.writer.add_scalars('Train_vs_Validation/Top5', {'train': t_top5, 'val': v_top5}, epoch)
    #
    #         perf_scoreboard.update(v_top1, v_top5, epoch)
    #         is_best = perf_scoreboard.is_best(epoch)
    #         if is_best:
    #             logger.info('!!!Checking how good in test set!!!')
    #             v_top1, v_top5, v_loss = process.validate(test_loader, model, criterion, epoch, monitors, args)
    #         util.save_checkpoint(epoch, args.arch, model, {'top1': v_top1, 'top5': v_top5}, is_best, args.name, log_dir)
    #
    #     logger.info('>>>>>>>> Epoch -1 (final model evaluation)')
    #     process.validate(test_loader, model, criterion, -1, monitors, args)

    # tbmonitor.writer.close()  # close the TensorBoard
    logger.info('Program completed successfully ... exiting ...')


if __name__ == "__main__":
    main()
