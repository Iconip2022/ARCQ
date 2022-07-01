import torch as t
import time
def grad_scale(x, scale):  # s s_grad_scale
    y = x
    y_grad = x * scale
    return (y - y_grad).detach() + y_grad  # 这一步是断离操作？本质上这里的结果s没变化


def round_pass(x):
    y = x.round()  # 这里确实是取整操作！！
    return (y - x).detach() + x  # 这里也有一个断离操作？




class Quantizer(t.nn.Module):
    def __init__(self, bit):
        super().__init__()

    def open_grad(self):
        pass
    def w_init_from(self, x, *args, **kwargs):
        pass

    def a_init_from(self, x, *args, **kwargs):
        pass

    def forward(self, x):
        raise NotImplementedError


class IdentityQuan(Quantizer):  # 作者给的样例的conv1的weight都是会来到这里。必须要求bit是None，否则报错。
    def __init__(self, bit=None, *args, **kwargs):
        super().__init__(bit)
        assert bit is None, 'The bit-width of identity quantizer must be None'
        self.sub_cri = lambda *args: 1
        self.norm_loss = 0
    def open_grad(self):
        pass

    def a_init_from(self, *args):
        pass
    def w_init_from(self, *args):
        pass

    def forward(self, x):  # 这其实是全精度的推理！！！
        self.norm_loss = 0  # identquant是不需要提供loss的
        return x


class LsqQuanA(Quantizer):  # 这玩意在前向传播计算s grad scale，然后进行了参数的量化操作！
    def __init__(self, bit, all_positive=False, symmetric=False, per_channel=True):
        super().__init__(bit)

        if all_positive:
            assert not symmetric, "Positive quantization cannot be symmetric"
            # unsigned activation is quantized to [0, 2^b-1]
            self.thd_neg = 0
            self.thd_pos = 2 ** bit - 1
        else:
            if symmetric:  # 对称的时候，会少掉一个映射位，比如8 position变为7 position
                # signed weight/activation is quantized to [-2^(b-1)+1, 2^(b-1)-1]
                self.thd_neg = - 2 ** (bit - 1) + 1
                self.thd_pos = 2 ** (bit - 1) - 1
            else:  # 非对称的时候，原汁原味，比如8 position的 -4~3
                # signed weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
                self.thd_neg = - 2 ** (bit - 1)
                self.thd_pos = 2 ** (bit - 1) - 1

        self.per_channel = per_channel  # 是否采取每个通道分配
        self.bit = bit  # scale默认的初始化。
    def a_init_from(self, act_list, *args, **kwargs):
        pass
    def forward(self, x):  # TODO前向计算的时候就把s grad scale求出来了？
        x_min = t.min(x)
        x_max = t.max(x)
        scale_a = (x_max-x_min)/(self.thd_pos-self.thd_neg)
        if abs(scale_a) < 0.0001:
            print('scale iI am 0:',scale_a)
            scale_a = 1
            zero_point = self.thd_neg - (x_min/scale_a).round()
        else:
            zero_point = self.thd_neg - x_min/scale_a.round()
        # zero_point = 0
        # q_x = (x / scale_a).round()
        derta_int = ((x / scale_a).round() - x/scale_a)*(2**self.bit)
        derta_int = derta_int.round()
        q_x = (x / scale_a).round() + zero_point # 量化参数的公式！可见不是永久量化，是需要推理的时候量化
        x = t.clamp(q_x, self.thd_neg, self.thd_pos)  # 当s为1时thread太大了，没啥影响！
        #　round操作，但是有断离，这里得到的确实是离散的整数值。
        # x = ( q_x - zero_point ) * scale_a  # 其实完成了反量化
        # derta_x = (q_x - x)/q_x
        return x,derta_int,zero_point,scale_a

class LsqQuanW(Quantizer):  # 这玩意在前向传播计算s grad scale，然后进行了参数的量化操作！
    def __init__(self, bit, all_positive=False, symmetric=False, per_channel=True):
        super().__init__(bit)

        if all_positive:
            assert not symmetric, "Positive quantization cannot be symmetric"
            # unsigned activation is quantized to [0, 2^b-1]
            self.thd_neg = 0
            self.thd_pos = 2 ** bit - 1
        else:
            if symmetric:  # 对称的时候，会少掉一个映射位，比如8 position变为7 position
                # signed weight/activation is quantized to [-2^(b-1)+1, 2^(b-1)-1]
                self.thd_neg = - 2 ** (bit - 1) + 1
                self.thd_pos = 2 ** (bit - 1) - 1
            else:  # 非对称的时候，原汁原味，比如8 position的 -4~3
                # signed weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
                self.thd_neg = - 2 ** (bit - 1)
                self.thd_pos = 2 ** (bit - 1) - 1

        self.per_channel = per_channel  # 这是用来初始化时区分全连接层和卷积层的。
        self.s = 1. #t.nn.Parameter(t.ones(1))  # scale默认的初始化。

    def w_init_from(self, x, *args, **kwargs): 
        pass # x是weight。套Parameter是因为可以进行grad
        # if self.per_channel:  # s再次初始化了，此时表示对每个通道都有一个s元素。隐性说明此时是关于卷积层(和通道相关,但是难免会变)的
        #     self.s = t.nn.Parameter(
        #         x.detach().abs().mean(dim=list(range(1, x.dim())), keepdim=True) * 2 / (self.thd_pos ** 0.5))
        # else:  # 隐性说明此时是关于全连接层的
        #     self.s = t.nn.Parameter(x.detach().abs().mean() * 2 / (self.thd_pos ** 0.5))

    def forward(self, x):  # TODO前向计算的时候就把s grad scale求出来了？
        
        x_min = t.min(x)
        # print(x_min)
        x_max = t.max(x)
        scale_a = (x_max-x_min)/(self.thd_pos-self.thd_neg)
        if abs(scale_a) < 0.0001:
            print('scale iI am 0:',scale_a)
            scale_a = 1
            zero_point = self.thd_neg - (x_min/scale_a).round()
        else:
            zero_point = self.thd_neg - x_min/scale_a.round()
        # zero_point = 0
        # q_x = (x / scale_a).round()
        derta_int = ((x / scale_a).round() - x/scale_a)*(2**self.bit)
        derta_int = derta_int.round()
        q_x = (x / scale_a).round() + zero_point # 量化参数的公式！可见不是永久量化，是需要推理的时候量化
        x = t.clamp(q_x, self.thd_neg, self.thd_pos) 
        
        return x,derta_int,zero_point,scale_a
