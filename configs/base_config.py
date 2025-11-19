# base configurations
model = dict(
    type='SegEarthSegmentation',
    # 'CLIP', 'BLIP', 'OpenCLIP', 'MetaCLIP', 'ALIP', 'SkyCLIP', 'GeoRSCLIP', 'RemoteCLIP'
    clip_type='CLIP',
    vit_type='ViT-B/16',      # 'ViT-B/16', 'ViT-L-14'
    model_type='SegEarth',   # 'vanilla', 'MaskCLIP', 'GEM', 'SCLIP', 'ClearCLIP', 'SegEarth'
    ignore_residual=True,
    feature_up=True,
    feature_up_cfg=dict(
        model_name='jbu_one',
    #     model_name='bilinear',
    #     model_path=None),
        # model_path='simfeatup_dev/weights/xclip_jbu_one_million_aid.ckpt'),
        model_path='simfeatup_dev/jbu_one/2gpu_forward_x_xclip_jbu_one_million_aid_attention_crf_0_tv_0.0_ent_0.0.ckpt'),
    cls_token_lambda=-0.3,
)
# model = dict(
#     type='SegEarthSegmentation',
#     # 'CLIP', 'BLIP', 'OpenCLIP', 'MetaCLIP', 'ALIP', 'SkyCLIP', 'GeoRSCLIP', 'RemoteCLIP'
#     clip_type='MobileCLIP2',
#     vit_type='S2',      # 'ViT-B/16', 'ViT-L-14',S0,S2
#     model_type='SegEarth',   # 'vanilla', 'MaskCLIP', 'GEM', 'SCLIP', 'ClearCLIP', 'SegEarth'
#     ignore_residual=True,
#     feature_up=True,
#     slide_crop=256,
#     slide_stride=128,
#     feature_up_cfg=dict(
#         # model_name='jbu_one_32',
#         model_name='bilinear',
#         # model_path = 'simfeatup_dev/jbu_32/bs8_patch32_train_norm_apple2clip-s0_jbu_one_32_million_aid_attention_crf_0_tv_0.0_ent_0.0.ckpt'),
#         model_path=None),
#     cls_token_lambda=-0.3,
# )
# model = dict(
#     type='CLIPSegWrapper',
#     model_name='CIDAS/clipseg-rd64-refined',
#     prob_thd=0.0,
#     logit_scale=50.0,
#     slide_crop=352,
#     slide_stride=176,
#     precompute_text=True,
#     clip_type='NA',
#     vit_type='NA',
#     model_type='NA',
#     use_template=True
# )

test_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])

default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, alpha=0.5, name='visualizer')
log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = None
resume = False

test_cfg = dict(type='TestLoop')

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=2000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', interval=1))
