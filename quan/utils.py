# from _typeshed import NoneType
from email.mime import base
from .quantizer import *
from .func import *
import logging
import xlwt
import torch as t
import math
import numpy as np
def quantizer(default_cfg, mod_name, this_cfg=None):  # default_cfg是默认的有关量化的全局配置，而this_cfg是特殊情况的量化配置
    '''
    :param default_cfg:
    :param this_cfg:
    :return:
    这个确实可以叫做量化器，因为其只是一个配置器，一些有关量化的规定在这，包括scale s
    '''
    target_cfg = dict(default_cfg)  # 进行类型转换
    if this_cfg is not None:  # 有特殊情况，考虑特殊情况
        for k, v in this_cfg.items():
            target_cfg[k] = v

    if target_cfg['bit'] is None:
        q = IdentityQuan  # 作者给的样例的conv1 weight都是会来到这里。必须要求bit是None，否则报错。作者的问题，默认32位了？可自行更改。
    elif target_cfg['mode'] == 'lsq':
        if mod_name == 'weight':
            q = LsqQuanW
        elif mod_name == 'activation':
            q = LsqQuanA
        else:
            raise ValueError('Cannot find mod `%s`')
    elif target_cfg['mode'] == 'lsqp':
        if mod_name == 'weight':
            q = LsqpQuanW
        elif mod_name == 'activation':
            q = LsqpQuanA
        else:
            raise ValueError('Cannot find mod `%s`')
    else:
        raise ValueError('Cannot find quantizer `%s`', target_cfg['mode'])

    target_cfg.pop('mode')
    return q(**target_cfg)

def find_modules_to_quantize(model, quan_scheduler, n_cout=0):  # 建立对应的量化层，到卷积层和激活层
    replaced_modules = dict()  # 记录的就是所有需要量化的层的量化层。方便以后替代。
    ori_modules = dict()
    replaced_module_names = []
    if n_cout == 0:
        mode_ = None
    elif n_cout%2 == 1:
        mode_ = 'update_x'
    else:
        mode_ = 'update_w'
    # n_cout = 9
    # mode_ = None

    # !resnet 18
    # !1024
    # scale_weight = [0.11219,0.09498,0.064449,0.051199,0.043768,0.051693,0.055355,0.093323,0.059132,0.049421,0.04352,0.03601,0.05172,0.033897,0.038861,0.026368,0.03253,0.03548,0.01928,0.03187]
    # weight_zeropoint = [0.00645,0.09428,0.07038,0.05283,0.04678,0.055404,0.07253,0.09399,0.06453,0.06037,0.04660,0.05223,0.05363,0.05262,0.05042,0.03385,0.04097,0.062,0.03537,0.03819]
    # # 100
    # scale_x = [0.02024,0.29778,0.14747,0.37655,0.15067,0.44009,0.16020,0.37427,0.21413,0.16559,0.27280,0.15079,0.25773,0.19441,0.14091,0.23610,0.12323,0.19947,0.39075,0.0092]
    
    #! resnet 50 初始化
    # 1,2,47,50
    # scale_weight = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
    # weight_zeropoint = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
    # scale_x = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]

    # resnet101  init 104  50 53
    scale_weight = []
    weight_zeropoint = []
    scale_x = []
    for i in range(53):
        scale_weight.append(0.01)
        weight_zeropoint.append(0.01)
        scale_x.append(0.01)



    # if n_cout == 0:
    #     # /code/matplot_mix/scale_/scale_50/resnet50_{}.csv 
    #     for i in range(53):
    #         tar_path = '/code/matplot_mix/scale_/resnet50/resnet50_{}.csv'.format(i)
    #         if os.path.exists(tar_path):
    #             with open(tar_path,'r') as f:
    #                 reader_ = csv.DictReader(f)
    #                 for cc in reader_:
    #                     # up_layer = cc['layer']
    #                     up_scale_x = cc['scale_x']
    #                     up_scale_w = cc['scale_w']
    #                     scale_weight[i] = float(up_scale_w)
    #                     weight_zeropoint[i] = float(cc['self.zeropoint'])
                        
    #                     scale_x[i] = float(up_scale_x)

    # elif mode_ == 'update_x' or mode_ == 'update_w':
    #     for i in range(53):
    #         tar_path = '/code/matplot_mix/scale_/resnet50/resnet50_{}.csv'.format(i)
    #         with open(tar_path,'r') as f:
    #             reader_ = csv.DictReader(f)
    #             for cc in reader_:
    #                 # up_layer = cc['layer']
    #                 up_scale_x = cc['scale_x']
    #                 up_scale_w = cc['scale_w']
    #                 scale_weight[i] = float(up_scale_w)
    #                 weight_zeropoint[i] = float(cc['self.zeropoint'])
    #                 scale_x[i] = float(up_scale_x)
    #         if i == 1:
    #             print('up_scale_w:{} up_scale_x:{}'.format(up_scale_w,up_scale_x))
    # else:
    #     pass
    # ! resnet 18
    # layer_choose = [1,5,3,8,0,10,18,7,13,12,15,6,2,11,4,9,14,17,16,19]
    # layer_choose = [1, 8, 2, 13, 3, 4, 6, 9, 5, 7, 10, 11, 18, 12, 14, 15, 16, 17, 19, 20]
    # layer_choose = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
    # layer_choose = [10, 15, 8, 12, 1, 20, 16, 19, 14, 9, 5, 7, 17, 11, 4, 3, 18, 2, 6, 13]
    # layer_choose = [] # 6[[]] 1,2,3,6,16,18 1,6,18 66.61 kong 60.9
    # layer_weight = [1,4,47,50] #,9
    # layer_choose.reverse()
    # mobile net
    # layer_choose = [1,50, 4,2,5,47, 8,  11, 6,12, 3, 9, 15, 7, 14, 18, 10, 21, 25, 22, 17, 24, 19, 16, 13, 23, 27, 31, 20, 28, 44, 35, 37, 34, 32, 38, 41, 40, 43, 26, 30, 29, 33, 51, 36, 39, 53, 46, 42, 48, 45, 49, 52]


    # # ! resnet 50
    # layer_choose = [4,1,50, 2,5,47, 8,  11, 6,12, 3, 9, 15, 7, 14, 18, 10, 21, 25, 22, 17, 24, 19, 16, 13, 23, 27, 31, 20, 28, 44, 35, 37, 34, 32, 38, 41, 40, 43, 26, 30, 29, 33, 51, 36, 39, 53, 46, 42, 48, 45, 49, 52]
    # # layer_choose = [50,1,4,47,11,8,5,2,16,24,9,21,14,6,18,15,12,25,3,43,34,37,40,22,31,19,27,28,44,48,35,29,32,7,38,51,10,41,13,26,46,17,52,45,20,30,42,23,49,36,39,33]
    # # layer_choose = [51, 53, 49, 45, 50, 28, 44, 46, 52, 47, 48, 26, 30, 31, 15, 27, 7, 25, 1, 33, 34, 39, 32, 36, 43, 17, 42, 35, 38, 5, 41, 40, 14, 29, 37, 3, 13, 18, 21, 23, 2, 12, 20, 24, 6, 10, 4, 16, 22, 19, 11, 8, 9, 54]
    # #2,47,50
    # # layer_choose = [53,52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    # last_choose = layer_choose[:19]
    # layer_weight = layer_choose
    # print(layer_weight)
    # last_choose.append(47)
    # last_choose.append(50)
    

    # layer_choose.reverse()
    # last_choose = layer_choose[:(20//2+1)]

    

    dict_name = {}
    base_cout = 0
    first_flag = 0
    weight_flag = 0
    cout = 0 
    # xinit_flag = 1
    # print('last_choose',last_choose)
    for name, module in model.named_modules():
        if type(module) in QuanModuleMapping.keys():  # 确认一下是不是允许量化的层
            if name in quan_scheduler.excepts:  # 再确认一下是不是特殊量化的层
                # if base_cout in last_choose: #base_cout == layer_choose[n_cout]: #
                #     first_flag = 1
                # if base_cout in layer_weight:
                #     weight_flag = 1
                replaced_modules[name] = QuanModuleMapping[type(module)](
                    module,
                    # quan_w_fn=[scale_weight[base_cout],weight_flag,weight_zeropoint[base_cout]],
                    quan_a_fn=[cout],
                    update_mode = mode_
                )
                # layer_choose.append(first_flag)
                first_flag = 0
                weight_flag = 0
                dict_name[base_cout] = name
                ori_modules[name] = module
            else:  # 是正常的需要量化的层
                # if  base_cout in last_choose: #base_cout == layer_choose[n_cout]:
                #     first_flag = 1
                # if base_cout in layer_weight:
                #     weight_flag = 1
                replaced_modules[name] = QuanModuleMapping[type(module)](
                    module,
                    # quan_w_fn=[scale_weight[base_cout],weight_flag,weight_zeropoint[base_cout]],
                    quan_a_fn=[cout],
                    update_mode = mode_
                )
                dict_name[base_cout] = name
                
                ori_modules[name] = module
                # first_flag = 0
                # weight_flag = 0
            if type(module) == list(QuanModuleMapping.keys())[0] and base_cout<len(scale_weight)-1:
                base_cout = base_cout + 1
                # if base_cout > 50:
                #     print(base_cout,type(module))
            replaced_module_names.append(name)
            cout += 1

        elif name in quan_scheduler.excepts:
            logging.warning('Cannot find module %s in the model, skip it' % name)
    # wb = xlwt.Workbook(encoding='ascii')
    # path_name = 'mobilenetv2_name'
    # ws = wb.add_sheet('mobilenetv2_name')
    # for key,value in dict_name.items():
    #     ws.write(0,key,key)
    #     ws.write(1,key,value)
    # wb.save('mxied_ex_{}.xls'.format(path_name))
    # wb.close()
    # print(cout)
    return replaced_modules
    #, ori_modules, replaced_module_names



def replace_module_by_names(model, modules_to_replace):  # 递归调用找，肥肠好用!
    def helper(child: t.nn.Module):
        for n, c in child.named_children():
            # 判断一下这里面有没有想找的类型
            if type(c) in QuanModuleMapping.keys():
                for full_name, m in model.named_modules():
                    if c is m:
                        # 虽然说是add添加，其实只要n名字重复了就会执行替换操作。
                        child.add_module(n, modules_to_replace[full_name])
                        # print('successful replace module!')
                        break
                #! edit by dongz
                # break
            else:
                helper(c)

    # 递归调用找，肥肠好用!
    helper(model)
    return model

def get_module_loss_by_names(model, modules_to_replace):  # 递归调用找，肥肠好用!
    def helper(child: t.nn.Module):
        for n, c in child.named_children():
            # 判断一下这里面有没有想找的类型
            if type(c) in QuanModuleMapping.keys():
                for full_name, m in model.named_modules():
                    if c is m:
                        # 虽然说是add添加，其实只要n名字重复了就会执行替换操作。
                        child.add_module(n, modules_to_replace[full_name])
                        print('successful replace module!')
                        break
            else:
                helper(c)

    # 递归调用找，肥肠好用!
    helper(model)
    return model

def quant_weight(model,per = 0.9):
    paralist = []
    paradict = {}
    score = []
    count = 0
    for name,module in model.named_modules():
        if 'QuanConv' in str(type(module)):
            count = count + 1
            num = sum(p.numel() for p in list(module.parameters()))
            paralist.append(num)
            paradict[name] = num
            # print(module) module.wucha module.wucha/(math.log(num)*math.sqrt(count))
            module.delta_score = module.wucha*(math.log(num)/math.sqrt(count))
            score.append(module.delta_score)
            

    numb = int(len(score)*(1-per))
    score.sort()
    print(score)
    minscore = score[int(len(score)*(1-per))]
    # print('len(score) {} len {} '.format(len(score),int(len(score)*per)))
    count_l = 0
    for name,module in model.named_modules():
        if 'QuanConv' in str(type(module)):
            if module.delta_score >= minscore:
                module.deltaflag = 1
                # print(count_l)
            count_l = count_l + 1
    print('countl',count_l)

            
