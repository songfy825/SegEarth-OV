from PIL import Image
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
# from segearth_segmentor import SegEarthSegmentation
from ss import SegEarthSegmentation
from CustomFastVit import CustomFastVit
from models.clipseg import CLIPSegWrapper
# img = Image.open('demo/oem_koeln_50.tif')
# img = Image.open('demo/1111.png')
img = Image.open('bird_full.jpg')

name_list = ['background', 'bareland,barren', 'grass', 'pavement', 'road',
             'tree,forest', 'water,river', 'cropland', 'building,roof,house','bird','flower']
class_colors = [
    [0, 0, 0],           # background - 黑色
    [139, 69, 19],       # bareland/barren - 棕色
    [0, 128, 0],         # grass - 绿色
    [128, 128, 128],     # pavement - 灰色
    [255, 255, 0],       # road - 黄色
    [0, 255, 0],         # tree/forest - 亮绿色
    [0, 0, 255],         # water/river - 蓝色
    [255, 255, 255],     # cropland - 白色
    [255, 0, 0],         # building/roof/house - 红色
    [255, 165, 0],        # bird - 橙色
    [0, 255, 255],        # flower - 青色
]

# 归一化到0-1范围
class_colors = np.array(class_colors) / 255.0

# 创建颜色映射
cmap = ListedColormap(class_colors)

with open('./configs/my_name.txt', 'w') as writers:
    for i in range(len(name_list)):
        if i == len(name_list)-1:
            writers.write(name_list[i])
        else:
            writers.write(name_list[i] + '\n')
writers.close()


# img_tensor = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
#     # transforms.Resize((224, 224))
# ])(img)

# img_tensor = img_tensor.unsqueeze(0).to('cuda')

# model = SegEarthSegmentation(
#     clip_type='CLIP',     # 'CLIP', 'BLIP', 'OpenCLIP', 'MetaCLIP', 'ALIP', 'SkyCLIP', 'GeoRSCLIP', 'RemoteCLIP'
#     vit_type='ViT-B/16',      # 'ViT-B/16', 'ViT-L-14'
#     model_type='SegEarth',   # 'vanilla', 'MaskCLIP', 'GEM', 'SCLIP', 'ClearCLIP', 'SegEarth'
#     ignore_residual=True,
#     feature_up=True,
#     feature_up_cfg=dict(
#         model_name='jbu_stack',
#         model_path='simfeatup_dev/weights/clip_jbu_stack_cocostuff.ckpt'),
#         # model_name='bilinear',
#         # model_path=None),
#     cls_token_lambda=-0.3,
#     name_path='./configs/my_name.txt',
#     prob_thd=0.1,
# )
# img_tensor = transforms.Compose([

#     transforms.ToTensor(),
#     # transforms.Resize((224, 224)),
# ])(img).to('cuda')
# img_tensor = img_tensor.unsqueeze(0).to('cuda')
# model = SegEarthSegmentation(
#     clip_type='MobileCLIP2',
#     vit_type='S0',      # 'ViT-B/16', 'ViT-L-14',S0,S1,S2
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
#     name_path='./configs/my_name.txt',
#     cls_token_lambda=-0.3,
# )
img_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((704, 704), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])(img)
img_tensor = img_tensor.unsqueeze(0).to('cuda')
model  = CLIPSegWrapper(
    model_name='CIDAS/clipseg-rd64-refined',
    name_path='configs/my_name.txt',  # 类别文件路径
    prob_thd=0.0,
    logit_scale=50.0,
    slide_crop=352,
    slide_stride=176,
    use_fp16=True,
    use_template=True
)
result = model.predict(img_tensor, data_samples=None)
# 处理返回的结果
if isinstance(result, list) and len(result) > 0:
    # 从返回的字典中提取预测结果
    if isinstance(result[0], dict):
        seg_pred = result[0]['pred_sem_seg'].cpu().numpy().squeeze(0)
    else:
        # 如果是 PixelData 对象
        seg_pred = result[0].cpu().numpy().squeeze(0)
else:
    # 如果直接返回了张量
    seg_pred = result.data.cpu().numpy().squeeze(0)

fig, ax = plt.subplots(1, 2, figsize=(15, 6))

# 显示原图
ax[0].imshow(img)
ax[0].set_title('Original Image')
ax[0].axis('off')

# 显示分割结果
im = ax[1].imshow(seg_pred, cmap=cmap, vmin=0, vmax=len(class_colors)-1)
ax[1].set_title('Segmentation Result')
ax[1].axis('off')

# 创建图例
classes = ['background', 'bareland', 'grass', 'pavement', 'road', 
           'tree', 'water', 'cropland', 'building','bird','flower']
colors = [cmap(i) for i in range(len(classes))]

# 添加图例
legend_elements = [plt.Line2D([0], [0], marker='s', color='w', 
                             markerfacecolor=colors[i], markersize=10, 
                             label=classes[i]) for i in range(len(classes))]
ax[1].legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))

plt.tight_layout()
# plt.savefig('clipseg.png', bbox_inches='tight', dpi=300)
plt.savefig('clipseg_wo_temp.png', bbox_inches='tight', dpi=300)
# plt.savefig('xclip.png', bbox_inches='tight', dpi=300)