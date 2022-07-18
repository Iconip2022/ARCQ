import torch as t
from torch import Tensor
import torch
from torch._C import ThroughputBenchmark
import torch.nn.functional as F
import time
from typing import Optional,List,Tuple,Union
import yaml
import logging
import csv 
import os,sys
import random
import numpy as np

def Get_Average(list_):
    sum = 0
    for item in list_:
        sum += item
    aver_ = sum/len(list_)
    max_ = max(list_)
    min_ = min(list_)
    clip_ = max(abs(min_-aver_),max_-aver_)*0.85

    
    last_list = list_.copy()
    for i in list_:
        if abs(i-aver_) > clip_ and abs(i-aver_) > 0.00001:
            last_list.remove(i)
    new_sum = 0
    for item in list_:
        new_sum += item

    return new_sum/len(list_)

def Get_Most(list_):

    list_set=set(list_)
    frequency_dict={}
    for i in list_set:
        frequency_dict[i]=list.count(i)# new_dict[key]=value
    grade_mode=[]
    for key,value in frequency_dict.items():
        if value==max(frequency_dict.values()):
            grade_mode.append(key)
    return grade_mode/len(grade_mode)


class QuanConv2d(t.nn.Conv2d):
    def __init__(self, m: t.nn.Conv2d, quan_w_fn=None, quan_a_fn=None,update_mode = None):
        """

        :param m:
        :param quan_w_fn:
        :param quan_a_fn:
        """
        assert type(m) == t.nn.Conv2d 
        super().__init__(m.in_channels, m.out_channels, m.kernel_size,
                         stride=m.stride,
                         padding=m.padding,
                         dilation=m.dilation,
                         groups=m.groups,
                         bias=True if m.bias is not None else False,
                         padding_mode=m.padding_mode)  
        
        self.weight = t.nn.Parameter(m.weight.detach())  
        if m.bias is not None:  
            self.bias = t.nn.Parameter(m.bias.detach())  
        # self.weight_zeropoint = t.tensor(0.0)
        self.eps = 0.00001
        self.update_scale = t.tensor(0.0)
        # rounding com flag 
        self.weight_residual_flag = 1#quan_w_fn[1]
        self.layer = quan_a_fn[0]
        # self.scale_x = quan_a_fn[0]
       
        self.update_mode = update_mode
        self.res_flag = 1#quan_a_fn[2]

        if self.layer == 0 :#
            self.bit = 8
            self.actbit = 8
        else:
            self.bit = 4
            self.actbit = 4
        
        self.neg = - 2 ** (self.bit - 1) 
        self.pos = 2 ** (self.bit - 1) - 1

        self.act_neg = - 2 ** (self.actbit - 1) 
        self.act_pos = 2 ** (self.actbit - 1) - 1
     
        self.cout = 0
        self.scale_updict = [[],[],[]]
        self.best_scale = 1.
        self.score_bestx = 0.1
        self.score_bestw = 0.1
        self.up_scale_x = 1.
        self.up_scale_w = 1.

        self.scale_wdict = [0,1,2,3]
        
        self.tar_path = '/code/matplot_mix/scale_/resnet50/resnet50_{}.csv'.format(self.layer)
        
        self.scalew_init()
        

    def scalew_init(self):
        w_max = t.max(self.weight)
        w_min = t.min(self.weight)
        # best_max = w_max
        # best_min = w_min

        
        best_score = 10000000
        count_n = 1
        # if self.weight_residual_flag == 0:
        # for i in range(count_n):
        #     new_min = w_min * (1.0 - (i * 0.005))
        #     new_max = w_max * (1.0 - (i * 0.005))
        #     # sym
        #     # scale_w = max(max(abs(new_min),abs(new_max))/self.pos,self.eps)
        #     # asy
        self.scale_w = max((w_max-w_min)/(self.pos-self.neg),self.eps)
        self.weight_zeropoint = (- w_min / self.scale_w).round()
            # # q_w = (self.weight / scale_w).round()+self.weight_zeropoint
            # q_w = t.clamp((self.weight / scale_w).round()+self.weight_zeropoint,0,self.pos-self.neg)
            # q_w = (q_w-self.weight_zeropoint)*scale_w
            # # self.quanweight = q_w
            # score = (self.weight - q_w).abs().pow(2.4).mean()
            # if score < best_score:
            #     best_score = score
            #     best_max = new_max
            #     best_min = new_min

        # sym
        # self.scale_w = max(max(abs(best_min),abs(best_max))/self.pos,self.eps)
        # asy
        # self.scale_w = max((best_max-best_min)/(self.pos-self.neg),self.eps)
        # self.weight_zeropoint = (- best_min / self.scale_w).round()
        w_derta_int = (-(self.weight / self.scale_w).round() + self.weight/self.scale_w)*(2**self.bit-1)
        
        
        # quanweight = (self.weight / self.scale_w).round() + self.weight_zeropoint
        quanweight = t.clamp((self.weight / self.scale_w).round() + self.weight_zeropoint,0,self.pos-self.neg)
        if self.weight_residual_flag and self.bit ==4:
            w_derta_int = w_derta_int.round()
            self.delta_w = w_derta_int.cuda()
        else:
            self.delta_w = 0*w_derta_int.cuda()

        self.quanweight = quanweight.cuda()
        self.requanweight = (self.quanweight- self.weight_zeropoint) * self.scale_w
        # self.scale_wdict[0] = self.quanweight
        # self.scale_wdict[1] = self.scale_w
        # self.scale_wdict[2] = self.delta_w
        # self.scale_wdict[3] = self.weight_zeropoint
        


    def update_scalew(self,weight,x_list):
        # x_list ： quant_x , scale_x , derta_x2int，x
        quant_x = x_list[0]
        scale_x = x_list[1]
        if self.res_flag == 1 and self.actbit == 4:
            derta_x2int = x_list[2]
        else:
            derta_x2int = 0*x_list[2]
        x = x_list[3]
        weight_max = t.max(weight)
        weight_min = t.min(weight)
        new_max = weight_max
        new_min = weight_min
        best_socre = 10000000
        best_wzeropoint = 0
        count_mse = 2
        output_ = self._conv_forward(x, weight,bias = None) 
        for i in range(count_mse):
            if i < count_mse//2:
                new_max = weight_max * (1 - 0.003*i)
            else:
                new_min = weight_min * (1 - 0.003*(i-count_mse//2))
            # quant sym
            # scale_w = max(max(abs(new_min),abs(new_max))/self.pos,self.eps)
            # asy
            scale_w = max((new_max-new_min)/(self.pos-self.neg),self.eps)
            self.weight_zeropoint = (- new_min / scale_w).round()

            # self.weight_zeropoint = t.tensor(0.0)
            # quant_weight = (weight / scale_w).round()+self.weight_zeropoint
            quant_weight = t.clamp((weight / scale_w).round()+self.weight_zeropoint,0,self.pos-self.neg)
            # weight residual
            weight_derta = (-(weight / scale_w).round() + weight/scale_w)*(2**self.bit-1)
            if self.weight_residual_flag and self.bit == 4:
                derta_w2int = weight_derta.round()
            else:
                derta_w2int = 0*weight_derta.round()
            # dequant only weight mse
            # q_w = (quant_weight-weight_zeropoint)*scale_w
            
            derta_float = (self._conv_forward(quant_x, derta_w2int, bias = None) + self._conv_forward(derta_x2int,quant_weight-self.weight_zeropoint,bias = None))/(2**self.bit-1)
            derta = derta_float.round()
            # res_round = y_q - self._conv_forward(z_x, self.quanweight,bias = None) #- self._conv_forward(quantized_act, z_w,bias = None) + self._conv_forward(z_x, z_w,bias = None)
            z_w = quant_weight * 0 + self.weight_zeropoint
            # quan conv 
           
            output_quan = self._conv_forward(quant_x, quant_weight,bias = None)  - self._conv_forward(quant_x, z_w,bias = None) 
            output_quan2float =( output_quan  + derta ) * scale_w*scale_x
            
             
            score = (output_ - output_quan2float).abs().pow(2.4).mean()
            if score < best_socre:
                best_socre = score
                bset_out = output_quan2float
                best_scalew = scale_w
                self.score_bestw = best_socre
                best_wzeropoint = self.weight_zeropoint
        self.weight_zeropoint = best_wzeropoint
        return best_scalew,bset_out
    
    def update_scalex(self,x,w_list):
        # w_list ： quant_w , scale_w , derta_w2int，w
        quant_weight = w_list[0]
        scale_w = w_list[1]
        if self.weight_residual_flag and self.bit == 4:
            derta_w2int = w_list[2]
        else:
            derta_w2int = 0*w_list[2]
        weight = self.weight
        x_max = t.max(x)
        x_min = t.min(x)
        new_max = x_max
        new_min = x_min
        best_socre = 10000000
        count_mse = 2
        output_ = self._conv_forward(x, weight,bias = None)  
        z_w = quant_weight * 0 + self.weight_zeropoint
        for i in range(count_mse):
            if i < count_mse//2:
                new_max = x_max * (1 - 0.001*i)
            else:
                new_min = x_min * (1 - 0.001*(i-count_mse//2))
            # quant 
            scale_x = max((new_max-new_min)/(self.act_pos-self.act_neg),self.eps)
            x_zeropoint = t.tensor(0.0)#(- new_min / scale_x).round()
            # quant_x = (x / scale_x).round()+ x_zeropoint
            quant_x = t.clamp((x / scale_x).round()+ x_zeropoint,self.act_neg,self.act_pos)
            z_x = quant_x*0 + x_zeropoint
            # weight residual
            x_derta = (-(x / scale_x).round() + x/scale_x)*(2**self.actbit-1)
            if self.res_flag == 1 and self.actbit == 4:
                derta_x2int =x_derta.round()
            else:
                derta_x2int =0*x_derta.round()
            # dequant only weight mse
            # q_w = (quant_weight-weight_zeropoint)*scale_w
           
            derta_float = self._conv_forward(quant_x-x_zeropoint, derta_w2int, bias = None)/(2**self.bit-1) + self._conv_forward(derta_x2int,quant_weight - self.weight_zeropoint,bias = None)/(2**self.actbit-1)
            derta = derta_float.round()
            
            # quan conv 
            
            
            output_quan = self._conv_forward(quant_x, quant_weight,bias = None)  - self._conv_forward(quant_x, z_w,bias = None)-self._conv_forward(z_x, quant_weight,bias = None) 

            output_quan2float = ( output_quan  + derta ) * scale_w*scale_x
            # float
            # output_quan = self._conv_forward(quant_x, quant_weight,bias = None)  
            score = (output_ - output_quan2float).abs().pow(2.4).mean()
            if score < best_socre:
                best_socre = score
                self.scale_x = scale_x
                self.score_bestx = best_socre
                besty = output_quan2float
        return self.scale_x,besty,x_zeropoint
    
    def _csv_read_write(self):
        self.scale_x = Get_Average(self.scale_updict[1])
        # self.zeropoint_x = Get_Average(self.scale_updict[2])
        # print('self.layer:{},self.best_scale:{:.5f}'.format(self.layer,self.best_scale))
        # tar_path = '/code/matplot_mix/scale_/scale_50/resnet50_{}.csv'.format(self.layer)
        # if not os.path.exists(self.tar_path):
        # #     os.makedirs(tar_path)
        #     with open(self.tar_path,'w') as csvfile:
        #         writer = csv.DictWriter(csvfile,fieldnames=['layer','scale_x','scale_w','self.zeropoint','self.score_bestx','self.score_bestw'])
        #         writer.writeheader()
        #        
        #         if not type(self.scale_x) == type(0.012):
        #             scale_x_write = float(self.scale_x.cpu().numpy())
        #         else:
        #             scale_x_write = self.scale_x
        #         if not type(self.scale_w) == type(0.012):
        #             scale_w_write = float(self.scale_w.cpu().numpy())
        #         else:
        #             scale_w_write = self.scale_w
        #         if not type(self.score_bestw) == type(0.012)  :
        #             score_bestw = float(self.score_bestw.cpu().numpy())
        #         else:
        #             score_bestw = self.score_bestw
        #         if not type(self.score_bestx) == type(0.012):
        #             score_bestx = float(self.score_bestx.cpu().numpy())
        #         else:
        #             score_bestx= self.score_bestx



        #         writer.writerow({'layer':str(self.layer),'scale_x':str(scale_x_write),'scale_w':str(scale_w_write),'self.zeropoint':str(weight_zeropoint_best),'self.score_bestx':str(score_bestx),'self.score_bestw':str(score_bestw)})


        # with open(self.tar_path,'r') as f:

        #     reader_ = csv.DictReader(f)
        #     for cc in reader_:
        #         up_layer = cc['layer']
        #         self.up_scale_x = cc['scale_x']
        #         self.up_scale_w = cc['scale_w']
                
        #         if self.update_mode == 'update_x':
        #             a = random.random()
        #             self.up_scale_x = str(self.best_scale*(0.5+a/2) + float(self.up_scale_x)*(1-a)/2 )
        #             # self.score_bestw = cc['self.score_bestw']
        #         elif self.update_mode == 'update_w':
        #             a = random.random()
        #             # self.score_bestx = cc['self.score_bestx']
        #             self.up_scale_w = str(self.best_scale*(0.5+a/2) + float(self.up_scale_w)*(1-a)/2 )
        #         else :
        #             pass
        #         if self.layer == 1:
        #             print(self.score_bestx)
        # with open(self.tar_path,'w') as csvfile:
        #     writer = csv.DictWriter(csvfile,fieldnames=['layer','scale_x','scale_w','self.zeropoint','self.score_bestx','self.score_bestw'])
        #     writer.writeheader()
        #     if not type(self.scale_x) == type(0.012):
        #             scale_x_write = float(self.scale_x.cpu().numpy())
        #     else:
        #         scale_x_write = self.scale_x
        #     if not type(self.scale_w) == type(0.012):
        #         scale_w_write = float(self.scale_w.cpu().numpy())
        #     else:
        #         scale_w_write = self.scale_w
        #     if not type(self.score_bestw) == type(0.012)  :
        #         score_bestw = float(self.score_bestw.cpu().numpy())
        #     else:
        #         score_bestw = self.score_bestw
        #     if not type(self.score_bestx) == type(0.012):
        #         score_bestx = float(self.score_bestx.cpu().numpy())
        #     else:
        #         score_bestx= self.score_bestx
        #     # for data in aba:
        #     writer.writerow({'layer':str(self.layer),'scale_x':str(self.up_scale_x),'scale_w':str(self.up_scale_w),'self.zeropoint':str(weight_zeropoint_best),'self.score_bestx':str(score_bestx),'self.score_bestw':str(score_bestw)})


            




    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, bias, self.stride_pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, x):
        # logging.basicConfig(filename='ptqtest.log',level=logging.DEBUG)
        
        
        if self.update_mode == 'update_x' and 0:
            self.scale_x,y,self.zeropoint_x = self.update_scalex(x,self.scale_wdict)
            self.update_scale = self.scale_x
        else:
            
            x_max = t.max(x)
            x_min = t.min(x)
            scale_x = max((x_max-x_min)/(self.act_pos-self.act_neg),self.eps)
            x_zeropoint = (- x_min / scale_x).round()
            # quant_x = (x / scale_x).round()+ x_zeropoint
            quant_x = t.clamp((x / scale_x).round()+ x_zeropoint,0,self.act_pos-self.act_neg)
            reqx = (quant_x-x_zeropoint)*scale_x
            # x_list = [0,1,2,3]
            # quant_x = x_list[0]
            # scale_x = x_list[1]
            # derta_x2int = x_list[2]
            # x = x_list[3]
        
            # x_list[0] =(x / self.scale_x).round()+ x_zeropoint
            # x_list[0] = t.clamp((x / self.scale_x).round()+ self.zeropoint_x,self.act_neg,self.act_pos)
            # x_list[1] = self.scale_x
            # x_derta = (-(x / scale_x).round() + x/scale_x)*(2**self.actbit-1)
            # if self.res_flag == 1 and self.actbit ==4:
            #     xdertaint = x_derta.round()
            # else:
            #     xdertaint = 0*x_derta.round()
            
            
                
                # self.weight_zeropoint = (- self.min_weight / self.scale_w).round()
                # quant_weight = (self.weight / self.scale_w).round() + self.weight_zeropoint
            # quant_weight = t.clamp((self.weight / self.scale_w).round() + self.weight_zeropoint,0,self.pos-self.neg)
                # weight residual
                # weight_derta = (-(self.weight / self.scale_w).round() + self.weight/self.scale_w)*(2**self.bit-1)
                # if self.weight_residual_flag and self.bit == 4:
                #     derta_w2int = weight_derta.round()
                # else:
                #     derta_w2int = 0*weight_derta.round()
            # z_x = quant_x*0 + x_zeropoint
            # z_w = self.quanweight * 0 + self.weight_zeropoint
            # dequant only weight mse
            # q_w = (quant_weight-weight_zeropoint)*scale_w
           
            # derta_float = self._conv_forward(quant_x, self.delta_w, bias = None)/(2**self.bit-1) + self._conv_forward(xdertaint,self.quanweight-self.weight_zeropoint,bias = None)/(2**self.actbit)
            # derta = derta_float.round()
            # output_quan = self._conv_forward(quant_x, self.quanweight,bias = None)  - self._conv_forward(quant_x, z_w,bias = None) - self._conv_forward(z_x, self.quanweight,bias = None)
            # y =( output_quan  + derta ) * self.scale_w*scale_x
        # if self.update_mode == None:
        y = self._conv_forward(reqx, self.requanweight,bias = self.bias)  
        #     score_ = (y-output_).abs().pow(2).mean()


        # self.cout += 1
        # if self.update_mode or not os.path.exists(self.tar_path) :

        #     if self.cout < 50: #10010
        #         # self.scale_updict[0].append(self.cout)
        #         # if type(self.zeropoint_x) == type(0.01):
        #         #     self.scale_updict[2].append(np.float16(self.zeropoint_x))
        #         # else:
        #         #     self.scale_updict[2].append(np.float16(self.zeropoint_x.cpu()))
        #         if type(self.update_scale) == type(0.01):
        #             self.scale_updict[1].append(np.float16(self.update_scale))
        #         else:
        #             self.scale_updict[1].append(np.float16(self.update_scale.cpu()))

        #     elif self.cout == 50:
        #         self._csv_read_write()
        #         if self.layer == 0 :
        #             print('{:.5}, mode:{}'.format(self.scale_x,self.update_mode))
            
        return y

class QuanLinear(t.nn.Linear):
    def __init__(self, m: t.nn.Linear, quan_w_fn=None, quan_a_fn=None,update_mode = None):
        assert type(m) == t.nn.Linear
        super().__init__(m.in_features, m.out_features,
                         bias=True if m.bias is not None else False)
        self.quan_w_fn = quan_w_fn
        self.quan_a_fn = quan_a_fn
        self.quan_b = quan_w_fn
        self.bit = 8
        self.neg = -2**(self.bit-1) 
        self.pos = 2**(self.bit-1) - 1
        self.eps = 0.00001
        self.weight = t.nn.Parameter(m.weight.detach())
        # self.quan_w_fn.w_init_from(m.weight)
        self.scalew_init()
        if m.bias is not None:  
            self.bias = t.nn.Parameter(m.bias.detach())
            # self.bias.data = self.quan_b(self.bias.data)
    
    def scalew_init(self):
        w_max = t.max(self.weight)
        w_min = t.min(self.weight)
        best_max = w_max
        best_min = w_min

        
        best_score = 10000000
        count_n = 10
        # if self.weight_residual_flag == 0:
        for i in range(count_n):
            new_min = w_min * (1.0 - (i * 0.005))
            new_max = w_max * (1.0 - (i * 0.005))
            # sym
            # scale_w = max(max(abs(new_min),abs(new_max))/self.pos,self.eps)
            # asy
            scale_w = max((new_max-new_min)/(self.pos-self.neg),self.eps)
            self.weight_zeropoint = (- new_min / scale_w).round()
            # q_w = (self.weight / scale_w).round()+self.weight_zeropoint
            q_w = t.clamp((self.weight / scale_w).round()+self.weight_zeropoint,0,self.pos-self.neg)
            q_w = (q_w-self.weight_zeropoint)*scale_w
            # self.quanweight = q_w
            score = (self.weight - q_w).abs().pow(2.4).mean()
            if score < best_score:
                best_score = score
                best_max = new_max
                best_min = new_min

        # sym
        # self.scale_w = max(max(abs(best_min),abs(best_max))/self.pos,self.eps)
        # asy
        self.scale_w = max((best_max-best_min)/(self.pos-self.neg),self.eps)
        self.weight_zeropoint = (- best_min / self.scale_w).round()
        # w_derta_int = (-(self.weight / self.scale_w).round() + self.weight/self.scale_w)*(2**self.bit-1)
        
        
        # quanweight = (self.weight / self.scale_w).round() + self.weight_zeropoint
        quanweight = t.clamp((self.weight / self.scale_w).round() + self.weight_zeropoint,0,self.pos-self.neg)
        # if self.weight_residual_flag and self.bit ==4:
        #     w_derta_int = w_derta_int.round()
        #     self.delta_w = w_derta_int.cuda()
        # else:
        #     self.delta_w = 0*w_derta_int.cuda()
        requan_weight = (quanweight-self.weight_zeropoint)*self.scale_w

        self.requanweight = requan_weight.cuda()
        
    def update_scalex(self,x,w_list):
        # w_list ： quant_w , scale_w , derta_w2int，w
        quant_weight = w_list[0]
        scale_w = w_list[1]
        if self.weight_residual_flag and self.bit == 4:
            derta_w2int = w_list[2]
        else:
            derta_w2int = 0*w_list[2]
        weight = self.weight
        x_max = t.max(x)
        x_min = t.min(x)
        new_max = x_max
        new_min = x_min
        best_socre = 10000000
        count_mse = 2
        output_ = self._conv_forward(x, weight,bias = None)  
        z_w = quant_weight * 0 + self.weight_zeropoint
        for i in range(count_mse):
            if i < count_mse//2:
                new_max = x_max * (1 - 0.0025*i)
            else:
                new_min = x_min * (1 - 0.0025*(i-count_mse//2))
            # quant 
            scale_x = max((new_max-new_min)/(self.pos-self.neg),self.eps)
            x_zeropoint = t.tensor(0.0)
            # quant_x = (x / scale_x).round()+ x_zeropoint
            quant_x = t.clamp((x / scale_x).round()+ x_zeropoint,self.neg,self.pos)
            # weight residual
            x_derta = (-(x / scale_x).round() + x/scale_x)*(2**self.bit-1)
            if self.res_flag == 1 and self.bit == 4:
                derta_x2int =x_derta.round()
            else:
                derta_x2int =0*x_derta.round()
            # dequant only weight mse
            # q_w = (quant_weight-weight_zeropoint)*scale_w
            
            derta_float = (self._conv_forward(quant_x, derta_w2int, bias = None) + self._conv_forward(derta_x2int,quant_weight - self.weight_zeropoint,bias = None))/(2**self.bit-1)
            derta = derta_float.round()
            
            # quan conv 
          
             
            output_quan = self._conv_forward(quant_x, quant_weight,bias = None)  - self._conv_forward(quant_x, z_w,bias = None) 

            output_quan2float = ( output_quan  + derta ) * scale_w*scale_x
            
            
            score = (output_ - output_quan2float).abs().pow(2.4).mean()
            if score < best_socre:
                best_socre = score
                self.scale_x = scale_x
                self.score_bestx = best_socre
                besty = output_quan2float
        return self.scale_x,besty
    
    def forward(self, x):
        x_max = t.max(x)
        x_min = t.min(x)
        # new_max = x_max
        # new_min = x_min
        scale_x = max((x_max-x_min)/(self.pos-self.neg),self.eps)
        x_zeropoint = t.tensor(0.0)
        quant_x = t.clamp((x / scale_x).round()+ x_zeropoint,self.neg,self.pos)
        requan_x = (quant_x-x_zeropoint)*scale_x
        # quantized_weight,w_delta_int,w_zeropoint,w_scale= self.quan_w_fn(self.weight)  
        # quantized_act,in_derta_int,in_zeropoint,in_scale = self.quan_a_fn(x)  
        return t.nn.functional.linear(requan_x, self.requanweight, self.bias) 


QuanModuleMapping = {
    t.nn.Conv2d: QuanConv2d,
    t.nn.Linear: QuanLinear
}
