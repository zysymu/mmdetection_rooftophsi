# dataset settings
dataset_type = 'RhsiDataset'
data_root = '/content/drive/MyDrive/SSHODC/'
train_pipeline = [
    dict(type='LoadMaskedHSIImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1600, 200), keep_ratio=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadMaskedHSIImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1600, 200),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='RandomFlip'),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=3,
        dataset=dict(
		type=dataset_type,
		ann_file=data_root + 'TRAIN/annotationsjson/instances_train_s10.json',
		img_prefix=data_root + 'TRAIN/spectral/',
		mask_prefix=data_root + 'TRAIN/masks/',
		pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'VAL/annotationsjson/instances_val_id.json',
        img_prefix=data_root + 'VAL/spectral/',
	mask_prefix=data_root + 'VAL/masks/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'VAL/annotationsjson/instances_val_id.json',
        img_prefix=data_root + 'VAL/spectral/',
	mask_prefix=data_root + 'VAL/masks/',
        pipeline=test_pipeline))
evaluation = dict(interval=2, metric='bbox', classwise=True)
