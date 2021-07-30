# -*- encoding: utf-8 -*-
"""
@File    : mobilt_pt.py
@Time    : 2021/7/26 17:47
@Author  : Zhl
@Desc    : pytorch-model -> torchscript
"""
from nets.pspnet import PSPNet as pspnet
import torch
import torchvision
from torch.utils.mobile_optimizer import optimize_for_mobile

model_path = 'model_data/best_model.pth'
net = pspnet(num_classes=2, downsample_factor=8, pretrained=True,
                  backbone="mobilenet_ca", aux_branch=False, skip_upsample=True)
state_dict = torch.load(model_path)
net.load_state_dict(state_dict, strict=False)
print('{} model, anchors, and classes loaded.'.format(model_path))

net = net.eval()

# 给模型的forward()方法一个示例输入
example = torch.rand(1, 3, 473, 473)

scripted_module = torch.jit.trace(net, example)
optimized_scripted_module = optimize_for_mobile(scripted_module)

# Export full jit version model (not compatible with lite interpreter)
scripted_module.save("model_data/mobile.pt")

# Export lite interpreter version model (compatible with lite interpreter)
scripted_module._save_for_lite_interpreter("model_data/mobile.ptl")

# using optimized lite interpreter model makes inference about 60% faster than the non-optimized lite interpreter model, which is about 6% faster than the non-optimized full jit model
optimized_scripted_module._save_for_lite_interpreter("model_data/mobile_optimized.ptl")


print("Finished Transformation")
