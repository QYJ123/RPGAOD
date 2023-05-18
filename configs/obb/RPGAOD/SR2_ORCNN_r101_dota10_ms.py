_base_ = './SR2_ORCNN_r50_dota10_ms.py'

# model
model = dict(pretrained='torchvision://resnet101', backbone=dict(depth=101))
