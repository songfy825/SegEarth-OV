import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmseg.models.segmentors import BaseSegmentor
from mmseg.models.data_preprocessor import SegDataPreProcessor
from mmengine.structures import PixelData
from mmseg.registry import MODELS
# from transformers import AutoProcessor, CLIPSegForImageSegmentation
from models.model_clipseg import CLIPSegForImageSegmentation
from models.clipseg_process import CLIPSegProcessor
from prompts.imagenet_template import *


@MODELS.register_module()
class CLIPSegWrapper(BaseSegmentor):
    """CLIPSeg open-vocabulary segmentation model with class text file support."""

    def __init__(self,
                 model_name='CIDAS/clipseg-rd64-refined',
                 name_path=None,
                 logit_scale=50.0,
                 prob_thd=0.0,
                 bg_idx=0,
                 slide_crop=None,
                 slide_stride=None,
                 use_fp16=True,
                 precompute_text=True,
                 use_template=False,
                 **kwargs):
        # 设置数据预处理器
        data_preprocessor = SegDataPreProcessor(
            mean=[127.5, 127.5, 127.5],  # 对应 (image - 0.5) / 0.5
            std=[127.5, 127.5, 127.5],   # 对应 (image - 0.5) / 0.5
            bgr_to_rgb=True,
            pad_val=0,
            seg_pad_val=255
        )
        super().__init__(data_preprocessor=data_preprocessor)

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        # 加载 CLIPSeg 模型
        # self.processor = AutoProcessor.from_pretrained(model_name)
        self.processor = CLIPSegProcessor.from_pretrained(model_name)
        self.model = CLIPSegForImageSegmentation.from_pretrained(
            model_name).to(self.device)
        self.model.eval()

        self.logit_scale = logit_scale
        self.prob_thd = prob_thd
        self.bg_idx = bg_idx
        self.slide_crop = slide_crop
        self.slide_stride = slide_stride
        self.use_fp16 = use_fp16
        self.precompute_text = precompute_text
        self.use_template = use_template
        # 加载类别文件
        assert name_path is not None, "You must provide cls_path to load class names."
        self.class_names, self.query_idx = self.get_cls_idx(name_path)
        self.num_classes = max(self.query_idx).item() + 1
        # 预计算文本嵌入
        if precompute_text:
            with torch.no_grad():
                # 使用模板处理文本
                if self.use_template:
                    # 应用模板并计算平均嵌入
                    text_embeddings = []
                    for class_name in self.class_names:
                        # 为每个类别名生成模板变体
                        # templated_texts = [
                        #     template(class_name) for template in openai_imagenet_template]
                        templated_texts = [
                            template(class_name) for template in ClipSeg_template]

                        # Tokenize模板变体
                        text_inputs = self.processor(
                            text=templated_texts,
                            return_tensors="pt",
                            padding=True
                        ).to(self.device)

                        # 获取文本特征
                        text_features = self.model.clip.get_text_features(
                            **text_inputs)

                        # 平均所有模板变体的特征
                        averaged_features = text_features.mean(
                            dim=0, keepdim=True)
                        # 归一化
                        averaged_features = averaged_features / \
                            averaged_features.norm(dim=-1, keepdim=True)
                        text_embeddings.append(averaged_features)

                    # 组合所有类别的嵌入
                    self.cached_text_inputs = torch.cat(text_embeddings, dim=0)

                else:
                    # 不使用模板，直接计算文本嵌入
                    self.cached_text_inputs = self.processor(
                        text=self.class_names,
                        return_tensors="pt",
                        padding=True
                    ).to(self.device)
        # # 可选预计算文本
        # if precompute_text:
        #     dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
        #     self.cached_text_inputs = self.processor(
        #         text=self.class_names,
        #         images=[dummy_img] * len(self.class_names),
        #         padding=True,
        #         return_tensors="pt",
        #     ).to(self.device)

    def get_cls_idx(self, path):
        with open(path, 'r') as f:
            name_sets = f.readlines()
        num_cls = len(name_sets)

        class_names, class_indices = [], []
        for idx in range(num_cls):
            names_i = name_sets[idx].split(',')
            class_names += [n.strip() for n in names_i]
            class_indices += [idx for _ in range(len(names_i))]

        # 将 class_indices 转换为 PyTorch 张量并保存到 self.query_idx
        self.query_idx = torch.tensor(
            class_indices, dtype=torch.long).to(self.device)
        return class_names, self.query_idx

    @torch.no_grad()
    def predict(self, inputs, data_samples):
        """统一预测接口"""
        if data_samples is not None:
            batch_img_metas = [
                data_sample.metainfo for data_sample in data_samples
            ]
        else:
            batch_img_metas = [
                dict(
                    ori_shape=inputs.shape[2:],
                    img_shape=inputs.shape[2:],
                    pad_shape=inputs.shape[2:],
                    padding_size=[0, 0, 0, 0])
            ] * inputs.shape[0]

        # inputs = inputs.half()
        if self.slide_crop > 0:
            seg_logits = self.forward_slide(
                inputs, batch_img_metas, self.slide_stride, self.slide_crop)
        else:
            seg_logits = self.forward_feature(
                inputs, batch_img_metas[0]['ori_shape'])

        return self.postprocess_result(seg_logits, data_samples)

    def forward_feature(self, img, logit_size=None):
        """前向推理特征提取"""
        if type(img) == list:
            img = img[0]

        if img.dim() == 3:
            img = img.unsqueeze(0)

        img = img.to(self.device)

        # 处理文本输入
        if self.use_template:
            text_inputs = {
                "conditional_embeddings": self.cached_text_inputs}
        else:
            if self.precompute_text:
                text_inputs = {k: v.clone().to(self.device)
                               for k, v in self.cached_text_inputs.items()}
                text_inputs.pop("pixel_values", None)
            else:
                text_inputs = self.processor(
                    text=self.class_names, padding=True, return_tensors="pt")
                text_inputs = {k: v.to(self.device)
                               for k, v in text_inputs.items()}

        with torch.cuda.amp.autocast(enabled=self.use_fp16):
            batch_imgs = img.repeat(len(self.class_names), 1, 1, 1)
            outputs = self.model(pixel_values=batch_imgs, **text_inputs)

        logits = outputs.logits.squeeze(1)

        # 调整输出尺寸
        if logit_size is None:
            logits = F.interpolate(logits.unsqueeze(0), size=img.shape[-2:],
                                   mode='bilinear', align_corners=False).squeeze(0)
        else:
            logits = F.interpolate(logits.unsqueeze(0), size=logit_size,
                                   mode='bilinear', align_corners=False).squeeze(0)

        return logits

    def forward_slide(self, img, img_metas, stride=176, crop_size=352):
        """滑动窗口推理实现"""
        if type(img) == list:
            img = img[0].unsqueeze(0)

        if type(stride) == int:
            stride = (stride, stride)
        if type(crop_size) == int:
            crop_size = (crop_size, crop_size)

        h_stride, w_stride = stride
        h_crop, w_crop = crop_size
        batch_size, _, h_img, w_img = img.shape
        out_channels = len(self.class_names)

        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1

        preds = torch.zeros(
            (batch_size, out_channels, h_img, w_img), device=self.device)
        count_mat = torch.zeros(
            (batch_size, 1, h_img, w_img), device=self.device)

        # 处理文本输入
        if self.use_template:
            # 封装成 键值对 conditional_embeddings: self.cached_text_inputs
            base_text_inputs = {
                "conditional_embeddings": self.cached_text_inputs}
        else:
            if self.precompute_text:
                base_text_inputs = {k: v.clone().to(self.device)
                                    for k, v in self.cached_text_inputs.items()}
                base_text_inputs.pop("pixel_values", None)
            else:
                base_text_inputs = self.processor(
                    text=self.class_names, padding=True, return_tensors="pt")
                base_text_inputs = {k: v.to(self.device)
                                    for k, v in base_text_inputs.items()}

        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)

                crop_img = img[:, :, y1:y2, x1:x2]

                # 处理需要padding的情况
                H, W = crop_img.shape[2:]
                pad = self.compute_padsize(H, W, 16)

                if any(pad):
                    crop_img = F.pad(crop_img, pad)

                # 推理
                with torch.cuda.amp.autocast(enabled=self.use_fp16):
                    batch_imgs = crop_img.repeat(
                        len(self.class_names), 1, 1, 1)
                    outputs = self.model(
                        pixel_values=batch_imgs, **base_text_inputs)
                    crop_seg_logit = outputs.logits.squeeze(1)

                # 移除padding
                if any(pad):
                    l, r, t, b = pad
                    crop_seg_logit = crop_seg_logit[:, t:t + H, l:l + W]

                # 调整到原始patch尺寸
                crop_seg_logit = F.interpolate(
                    crop_seg_logit.unsqueeze(0),
                    size=(y2 - y1, x2 - x1),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)

                preds[:, :, y1:y2, x1:x2] += crop_seg_logit
                count_mat[:, :, y1:y2, x1:x2] += 1

        assert (count_mat == 0).sum() == 0
        preds = preds / count_mat

        # 调整到原始图像尺寸
        img_size = img_metas[0]['ori_shape'][:2]
        logits = F.interpolate(preds, size=img_size, mode='bilinear')

        return logits

    def compute_padsize(self, H: int, W: int, patch_size: int):
        """计算需要填充的尺寸以使图像尺寸能被patch_size整除"""
        l, r, t, b = 0, 0, 0, 0
        if W % patch_size:
            lr = patch_size - (W % patch_size)
            l = lr // 2
            r = lr - l

        if H % patch_size:
            tb = patch_size - (H % patch_size)
            t = tb // 2
            b = tb - t

        return l, r, t, b

    def postprocess_result(self, seg_logits, data_samples):
        batch_size = seg_logits.shape[0]
        for i in range(batch_size):
            seg_logits = seg_logits[i] * self.logit_scale
            seg_logits = seg_logits.softmax(0)  # n_queries * w * h

            num_cls, num_queries = max(self.query_idx) + 1, len(self.query_idx)
            if num_cls != num_queries:
                seg_logits = seg_logits.unsqueeze(0)
                cls_index = F.one_hot(self.query_idx)
                cls_index = cls_index.T.view(num_cls, num_queries, 1, 1)
                seg_logits = (seg_logits * cls_index).max(1)[0]

            seg_pred = seg_logits.argmax(0, keepdim=True)
            seg_pred[seg_logits.max(0, keepdim=True)[0]
                     < self.prob_thd] = self.bg_idx

            if data_samples is None:
                return seg_pred
            else:
                data_samples[i].set_data({
                    'seg_logits': PixelData(**{'data': seg_logits}),
                    'pred_sem_seg': PixelData(**{'data': seg_pred})
                })
        return data_samples

    def _forward(data_samples):
        """
        """

    def inference(self, img, batch_img_metas):
        """
        """

    def encode_decode(self, inputs, batch_img_metas):
        """
        """

    def extract_feat(self, inputs):
        """
        """

    def loss(self, inputs, data_samples):
        """
        """
