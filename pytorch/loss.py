import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F
import pdb


def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy


def grl_hook(coeff):
    def fun1(grad):
        return -coeff * grad.clone()

    return fun1


def CDAN(input_list, ad_net, entropy=None, coeff=None, random_layer=None):
    # 本质上就是用 encoder 的features 与 softmax的features叉乘，作为映射。
    softmax_output = input_list[1].detach()
    feature = input_list[0]
    if random_layer is None:
        # 提取的特征与特征经过全连接后的输出然后作softmax变换后 的概率表达做一个外积，这个结果再放到discriminator里面进行判断，labels就是1和0
        # 这里和算欧式距离一个意思，softmax概率扩展  乘以 特征维第一维  (列数1等于行数1)
        op_out = torch.bmm(softmax_output.unsqueeze(2), feature.unsqueeze(1))
        # 这里就是输出 映射，得到[batch_size, num_classes, feature_dim]
        # 计算两个tensor的矩阵乘法，torch.bmm(a,b),tensor a 的size为(b,h,w),tensor b的size为(b,w,m)
        # 也就是说两个tensor的第一维是相等的，然后第一个数组的第三维和第二个数组的第二维度要求一样，对于剩下的则不做要求，输出维度 （b,h,m）

        ad_out = ad_net(op_out.view(-1, softmax_output.size(1) * feature.size(1)))
        # 注意得到后是已经梯度翻转的了，不过也不影响后面哈
    else:
        random_out = random_layer.forward([feature, softmax_output])
        ad_out = ad_net(random_out.view(-1, random_out.size(1)))
    batch_size = softmax_output.size(0) // 2
    # 定义domain label
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().cuda()
    if entropy is not None:
        # 为了应对那些对迁移有负面影响的样本，用熵来控制它们的重要性，把熵操作加到了对抗网络中。
        # 利用熵来给每个训练样本赋予不同的权重
        entropy.register_hook(grl_hook(coeff))
        entropy = 1.0 + torch.exp(-entropy)
        source_mask = torch.ones_like(entropy)
        source_mask[feature.size(0) // 2:] = 0
        source_weight = entropy * source_mask
        target_mask = torch.ones_like(entropy)
        target_mask[0:feature.size(0) // 2] = 0
        target_weight = entropy * target_mask
        weight = source_weight / torch.sum(source_weight).detach().item() + \
                 target_weight / torch.sum(target_weight).detach().item()
        return torch.sum(weight.view(-1, 1) * nn.BCELoss(reduction='none')(ad_out, dc_target)) / torch.sum(weight).detach().item()
    else:
        return nn.BCELoss()(ad_out, dc_target)


def DANN(features, ad_net):
    ad_out = ad_net(features)
    batch_size = ad_out.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().cuda()
    return nn.BCELoss()(ad_out, dc_target)
