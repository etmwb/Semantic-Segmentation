import torch
import torch.nn as nn
from torch.autograd import gradcheck

from detseg.ops import (DeformConv, DeformConvPack, ModulatedDeformConv,
                        ModulatedDeformConvPack, DepthawareConv, DepthDeformConvPack,
                        deform_conv, modulated_deform_conv)


deformable_groups = 1
batch_size, in_channels, image_h, image_w = 2, 4, 4, 4
out_channels = 4
kernel_h, kernel_w = 3, 3


torch.manual_seed(3)
def check_dconv_zero_offset():
    conv_offset = nn.Conv2d(in_channels, deformable_groups * 2 * kernel_h * kernel_w,
                            kernel_size=(kernel_h, kernel_w),
                            stride=(1, 1),
                            padding=(1, 1),
                            bias=True).cuda()

    dcn = DeformConv(in_channels, out_channels, (kernel_h, kernel_w),
                   stride=1, padding=1, dilation=1,
                   groups=2,
                   deformable_groups=deformable_groups,
                   bias=True).cuda()
    pcn = nn.Conv2d(in_channels, out_channels, (kernel_h, kernel_w), stride=1, padding=1, dilation=1, groups=2).cuda()
    pcn.weight = dcn.weight
    pcn.bias = dcn.bias
    print((pcn.weight.data - dcn.weight.data).abs().max())

    conv_offset.weight.data.zero_()
    conv_offset.bias.data.zero_()
    # conv_identify(dcn.weight, dcn.bias)

    input = torch.randn(batch_size, in_channels, image_h, image_w).cuda()
    offset = conv_offset(input)
    output_d = dcn(input, offset)
    output_p = pcn(input)
    d = (output_d - output_p).abs().max()
    if d < 1e-5:
        print('dconv zero offset passed with {}'.format(d))
    else:
        print('dconv zero offset failed with {}'.format(d))
        # print(output_p)
        # print(output_d)
        print((output_d - output_p).abs())


def check_mdconv_zero_offset():
    conv_offset = nn.Conv2d(in_channels, deformable_groups * 2 * kernel_h * kernel_w,
                            kernel_size=(kernel_h, kernel_w),
                            stride=(1, 1),
                            padding=(1, 1),
                            bias=True).cuda()

    conv_mask = nn.Conv2d(in_channels, deformable_groups * 1 * kernel_h * kernel_w,
                          kernel_size=(kernel_h, kernel_w),
                          stride=(1, 1),
                          padding=(1, 1),
                          bias=True).cuda()

    dcn = ModulatedDeformConv(in_channels, out_channels, (kernel_h, kernel_w),
                   stride=1, padding=1, dilation=1,
                   groups=2,
                   deformable_groups=deformable_groups,
                   bias=True).cuda()
    pcn = nn.Conv2d(in_channels, out_channels, (kernel_h, kernel_w), stride=1, padding=1, dilation=1, groups=2).cuda()
    pcn.weight = dcn.weight
    pcn.bias = dcn.bias
    print((pcn.weight.data - dcn.weight.data).abs().max())

    conv_offset.weight.data.zero_()
    conv_offset.bias.data.zero_()
    conv_mask.weight.data.zero_()
    conv_mask.bias.data.zero_()

    input = torch.randn(batch_size, in_channels, image_h, image_w).cuda()
    offset = conv_offset(input)
    mask = conv_mask(input)
    mask = torch.sigmoid(mask)
    mask *= 2
    output_d = dcn(input, offset, mask)
    output_p = pcn(input)
    d = (output_d - output_p).abs().max()
    if d < 1e-5:
        print('mdconv zero offset passed with {}'.format(d))
    else:
        print('mdconv zero offset failed with {}'.format(d))
        # print(output_p)
        # print(output_d)
        print((output_d - output_p).abs())


def conv_identify(weight, bias, groups=1):
    weight.data.zero_()
    bias.data.zero_()
    o, i, h, w = weight.shape
    y = h//2
    x = w//2
    oc = o // groups
    for p in range(i):
        for q in range(o):
            if (p) == (q % oc):
                # print(q, p, y, x)
                # print(q % oc)
                weight.data[q, p, y, x] = 1.0


def check_dconv_zero_offset_identify():
    conv_offset = nn.Conv2d(in_channels, deformable_groups * 2 * kernel_h * kernel_w,
                            kernel_size=(kernel_h, kernel_w),
                            stride=(1, 1),
                            padding=(1, 1),
                            bias=True).cuda()

    groups = 2
    dcn = DeformConv(in_channels, out_channels, (kernel_h, kernel_w),
        stride=1, padding=1, dilation=1,
        groups=groups,
        deformable_groups=deformable_groups,
        bias=True).cuda()

    conv_offset.weight.data.zero_()
    conv_offset.bias.data.zero_()
    conv_identify(dcn.weight, dcn.bias, groups)

    input = torch.randn(batch_size, in_channels, image_h, image_w).cuda()
    offset = conv_offset(input)
    output = dcn(input, offset)
    d = (input - output).abs().max()
    if d < 1e-10:
        print('dconv zero offset identify passed with {}'.format(d))
    else:
        print('dconv zero offset identify failed with {}'.format(d))
        # print(input)
        # print(output)
        print((input - output).abs())

def check_mdconv_zero_offset_identify():
    conv_offset = nn.Conv2d(in_channels, deformable_groups * 2 * kernel_h * kernel_w,
                            kernel_size=(kernel_h, kernel_w),
                            stride=(1, 1),
                            padding=(1, 1),
                            bias=True).cuda()

    conv_mask = nn.Conv2d(in_channels, deformable_groups * 1 * kernel_h * kernel_w,
                          kernel_size=(kernel_h, kernel_w),
                          stride=(1, 1),
                          padding=(1, 1),
                          bias=True).cuda()

    groups = 2
    dcn = ModulatedDeformConv(in_channels, out_channels, (kernel_h, kernel_w),
        stride=1, padding=1, dilation=1,
        groups=groups,
        deformable_groups=deformable_groups,
        bias=True).cuda()

    conv_offset.weight.data.zero_()
    conv_offset.bias.data.zero_()
    conv_mask.weight.data.zero_()
    conv_mask.bias.data.zero_()
    conv_identify(dcn.weight, dcn.bias, groups)

    input = torch.randn(batch_size, in_channels, image_h, image_w).cuda()
    offset = conv_offset(input)
    mask = conv_mask(input)
    mask = torch.sigmoid(mask)
    output = dcn(input, offset, mask)
    output *= 2
    d = (input - output).abs().max()
    if d < 1e-10:
        print('mdconv zero offset identify passed with {}'.format(d))
    else:
        print('mdconv zero offset identify failed with {}'.format(d))
        # print(input)
        # print(output)
        print((input - output).abs())


def check_gradient_conv():

    input = torch.rand(batch_size, in_channels, image_h, image_w).double().cuda() * 0.01
    input.requires_grad = True
    from torch.nn.functional import conv2d

    weight = torch.randn(out_channels, in_channels, kernel_h, kernel_w).double().cuda()
    weight.requires_grad = True

    bias = torch.rand(out_channels).double().cuda()
    bias.requires_grad = True

    stride = 1
    padding = 1
    dilation = 1

    # print('check_gradient_conv: ',
    #       gradcheck(conv2d, (input, weight, bias,
    #                 stride, padding, dilation, deformable_groups),
    #                 eps=1e-3, atol=1e-2, rtol=1e-2, raise_exception=True))
    print('check_gradient_conv: ',
          gradcheck(conv2d, (input, weight, bias,
                    stride, padding, dilation, deformable_groups)))

def check_gradient_dconv():

    stride = 1
    padding = 1
    groups = 2
    dilation = 1
    im2col_step = 1

    input = torch.rand(batch_size, in_channels, image_h, image_w).double().cuda() * 0.01
    input.requires_grad = True

    offset = torch.randn(batch_size, deformable_groups * 2 * kernel_h * kernel_w, image_h, image_w).double().cuda() * 2
    # offset.data.zero_()
    # offset.data -= 0.5
    offset.requires_grad = True

    weight = torch.randn(out_channels, int(in_channels//groups), kernel_h, kernel_w).double().cuda()
    weight.requires_grad = True

    bias = torch.rand(out_channels).double().cuda()
    bias.requires_grad = True

    print('check_gradient_dconv: ',
          gradcheck(deform_conv, (input, offset, weight, bias,
                    stride, padding, dilation, groups, deformable_groups),
                    eps=1e-3, atol=1e-3, rtol=1e-2, raise_exception=True))
    # print('check_gradient_dconv: ',
    #       gradcheck(_DeformConv, (input, offset, weight, bias,
    #                 stride, padding, dilation, deformable_groups)))

def check_gradient_mdconv():
    stride = 1
    padding = 1
    groups = 2
    dilation = 1
    im2col_step = 1

    input = torch.rand(batch_size, in_channels, image_h, image_w).double().cuda() * 0.01
    input.requires_grad = True

    offset = torch.randn(batch_size, deformable_groups * 2 * kernel_h * kernel_w, image_h, image_w).double().cuda() * 2
    # offset.data.zero_()
    # offset.data -= 0.5
    offset.requires_grad = True

    mask = torch.rand(batch_size, deformable_groups * 1 * kernel_h * kernel_w, image_h, image_w).double().cuda()
    # mask.data.zero_()
    mask.requires_grad = True
    mask = torch.sigmoid(mask)

    weight = torch.randn(out_channels, int(in_channels//groups), kernel_h, kernel_w).double().cuda()
    weight.requires_grad = True

    bias = torch.rand(out_channels).double().cuda()
    bias.requires_grad = True

    print('check_gradient_mdconv: ',
          gradcheck(modulated_deform_conv, (input, offset, mask, weight, bias,
                    stride, padding, dilation, groups, deformable_groups),
                    eps=1e-3, atol=1e-3, rtol=1e-2, raise_exception=True, nondet_tol=1.0))


if __name__ == '__main__':
    check_dconv_zero_offset()
    check_mdconv_zero_offset()
    check_dconv_zero_offset_identify()
    check_mdconv_zero_offset_identify()
    check_gradient_conv()
    check_gradient_dconv()
    check_gradient_mdconv()