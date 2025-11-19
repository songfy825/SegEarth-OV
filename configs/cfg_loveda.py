_base_ = './base_config.py'

# model settings
model = dict(
    name_path='./configs/cls_loveda.txt',
    prob_thd=0.3,
)

# dataset settings
# Category labels: background – 1, building – 2, road – 3, water – 4, barren – 5,forest – 6, agriculture – 7. 
# And the no-data regions were assigned 0 which should be ignored. 
dataset_type = 'LoveDADataset'
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
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        reduce_zero_label=True,
        data_prefix=dict(
            img_path='loveDA/img_dir/val',
            seg_map_path='loveDA/ann_dir/val'),
        pipeline=test_pipeline))