_base_ = './base_config.py'

# model settings
model = dict(
    name_path='./configs/cls_vdd.txt',
    prob_thd=0.3,
)

# dataset settings
dataset_type = 'VDDDataset'
data_root = '/home/fengyi/data/SegEarth/'

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(448, 448), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]
# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='Resize', scale=(512, 512), keep_ratio=True),
#     # add loading annotation after ``Resize`` because ground truth
#     # does not need to do resize data transform
#     dict(type='LoadAnnotations'),
#     dict(type='PackSegInputs')
# ]
#Clipseg
# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='Resize', scale=(704, 704), keep_ratio=True),
#     # 添加数据类型转换和归一化
#     dict(type='LoadAnnotations'),
#     dict(type='PackSegInputs')
# ]
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='VDD/test/src',
            seg_map_path='VDD/test/gt'),
        pipeline=test_pipeline))