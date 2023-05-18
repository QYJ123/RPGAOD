_base_ = './ORCNN_r50_hrsc.py'

# model
model = dict(pretrained='torchvision://resnet101', backbone=dict(depth=101))

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[24, 38])
total_epochs = 50

