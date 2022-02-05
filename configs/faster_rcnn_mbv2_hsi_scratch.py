_base_ = [
	'_base_/models/faster_rcnn_mbv2_fpn.py',
	'_base_/datasets/rooftophsi_hsi.py',
	'_base_/schedules/schedule_scratch.py',
	'_base_/default_runtime.py'
]

norm_cfg=dict(type='BN', requires_grad=True)
model = dict(
    pretrained=None,
    backbone=dict(
	type='MobileNetV2HSI'),
    neck=dict(norm_cfg=norm_cfg),
    roi_head=dict(
        bbox_head=dict(
            type='Shared4Conv1FCBBoxHead',
            conv_out_channels=256,
            norm_cfg=norm_cfg)))
