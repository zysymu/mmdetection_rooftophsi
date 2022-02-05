# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001,paramwise_cfg=dict(norm_decay_mult=0))
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.1,
    step=[10, 20, 25])
runner = dict(type='EpochBasedRunner', max_epochs=30)

