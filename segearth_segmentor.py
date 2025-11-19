from simfeatup_dev.upsamplers import get_upsampler
import gem
from BLIP.models.blip_retrieval import blip_retrieval
from open_clip import tokenizer, create_model, get_tokenizer, create_model_from_pretrained
import torch.nn.functional as F
from mmseg.registry import MODELS
from mmengine.structures import PixelData
from mmseg.models.data_preprocessor import SegDataPreProcessor
from mmseg.models.segmentors import BaseSegmentor
from prompts.imagenet_template import *
from simfeatup_dev.fuse import *
from PIL import Image
import torch
import torch.nn as nn
import sys
import copy
import logging
sys.path.append("..")

logging.basicConfig(level=logging.INFO)
@MODELS.register_module()
class SegEarthSegmentation(BaseSegmentor):
    def __init__(self,
                 clip_type,
                 vit_type,
                 model_type,
                 name_path,
                 device=torch.device('cuda'),
                 ignore_residual=True,
                 prob_thd=0.0,
                 logit_scale=50,
                 slide_stride=256,
                 slide_crop=128,
                 cls_token_lambda=0,
                 bg_idx=0,
                 feature_up=True,
                 feature_up_cfg=dict(
                     model_name='jbu_one',
                     model_path='your/model/path')):
        if clip_type not in ['MobileCLIP', 'MobileCLIP2']:
            data_preprocessor = SegDataPreProcessor(
                mean=[122.771, 116.746, 104.094],
                std=[68.501, 66.632, 70.323],
                bgr_to_rgb=True)
            super().__init__(data_preprocessor=data_preprocessor)
        else:
            super().__init__()  # 不使用预处理器
        if clip_type == 'CLIP':
            if 'B' in vit_type:
                self.net = create_model(
                    'ViT-B/16', pretrained='openai', precision='fp16')
            elif 'L' in vit_type:
                self.net = create_model(
                    'ViT-L-14', pretrained='openai', precision='fp16')
        elif clip_type == 'RemoteCLIP':
            if 'B' in vit_type:
                self.net = create_model(
                    'ViT-B/32', pretrained='checkpoint/RemoteCLIP-ViT-B-32.pt', precision='fp16')
            elif 'L' in vit_type:
                self.net = create_model(
                    'ViT-L-14', pretrained='checkpoint/RemoteCLIP-ViT-L-14.pt', precision='fp16')
        elif clip_type == 'GeoRSCLIP':
            if 'B' in vit_type:
                self.net = create_model(
                    'ViT-B/32', pretrained='checkpoint/RS5M_ViT-B-32.pt', precision='fp16')
            elif 'L' in vit_type:
                self.net = create_model(
                    'ViT-L-14', pretrained='checkpoint/RS5M_ViT-L-14.pt', precision='fp16')
            elif 'H' in vit_type:
                self.net = create_model(
                    'ViT-H-14', pretrained='checkpoint/RS5M_ViT-H-14.pt', precision='fp16')
        elif clip_type == 'SkyCLIP':
            if 'B' in vit_type:
                self.net = create_model('ViT-B/32',
                                        pretrained='checkpoint/SkyCLIP_ViT_B32_top50pct/epoch_20.pt',
                                        precision='fp16')
            elif 'L' in vit_type:
                self.net = create_model('ViT-L-14',
                                        pretrained='checkpoint/SkyCLIP_ViT_L14_top30pct_filtered_by_CLIP_laion_RS/epoch_20.pt',
                                        precision='fp16')
        elif clip_type == 'OpenCLIP':
            if 'B' in vit_type:
                self.net = create_model(
                    'ViT-B/16', pretrained='laion2b_s34b_b88k', precision='fp16')
            elif 'L' in vit_type:
                self.net = create_model(
                    'ViT-L-14', pretrained='laion2b_s32b_b82k', precision='fp16')
        elif clip_type == 'MetaCLIP':
            if 'B' in vit_type:
                self.net = create_model(
                    'ViT-B-16-quickgelu', pretrained='metaclip_fullcc', precision='fp16')
            elif 'L' in vit_type:
                self.net = create_model(
                    'ViT-L/14-quickgelu', pretrained='metaclip_fullcc', precision='fp16')
        elif clip_type == 'BLIP':
            if 'B' in vit_type:
                self.net = blip_retrieval(
                    pretrained='checkpoint/model_base_14M.pth', image_size=slide_crop, vit='base')
            elif 'L' in vit_type:
                self.net = blip_retrieval(
                    pretrained='checkpoint/model_large.pth', image_size=slide_crop, vit='large')
            self.net = self.net.half()
        elif clip_type == 'ALIP':
            self.net = create_model(
                'ViT-B/32', pretrained='checkpoint/ALIP_YFCC15M_B32.pt', precision='fp16')
        elif clip_type == 'MobileCLIP2':  # MobileCLIP2-S0
            if vit_type == 'S0':
                self.net, self.preprocess = create_model_from_pretrained(
                    'MobileCLIP2-S0', pretrained="dfndr2b", precision='fp16')
            if vit_type == 'S2':
                self.net, self.preprocess = create_model_from_pretrained(
                    'MobileCLIP2-S2', pretrained="dfndr2b", precision='fp16')
        if model_type == 'GEM':
            if 'B' in vit_type:
                if clip_type == 'CLIP':
                    self.net = gem.create_gem_model(
                        'ViT-B/16', 'openai', ignore_residual=ignore_residual, device=device, precision='fp16')
                elif clip_type == 'OpenCLIP':
                    self.net = gem.create_gem_model(
                        'ViT-B/16', 'laion2b_s34b_b88k', ignore_residual=ignore_residual, device=device, precision='fp16')
                elif clip_type == 'MetaCLIP':
                    self.net = gem.create_gem_model(
                        'ViT-B/16-quickgelu', 'metaclip_fullcc', ignore_residual=ignore_residual, device=device, precision='fp16')
            elif 'L' in vit_type:
                if clip_type == 'CLIP':
                    self.net = gem.create_gem_model(
                        'ViT-L-14', 'openai', ignore_residual=ignore_residual, device=device, precision='fp16')
                elif clip_type == 'OpenCLIP':
                    self.net = gem.create_gem_model(
                        'ViT-L-14', 'laion2b_s32b_b82k', ignore_residual=ignore_residual, device=device, precision='fp16')
                elif clip_type == 'MetaCLIP':
                    self.net = gem.create_gem_model(
                        'ViT-L-14-quickgelu', 'metaclip_fullcc', ignore_residual=ignore_residual, device=device, precision='fp16')
            self.net = self.net.model
        self.device = device
        self.net.eval().to(device)
        if clip_type in ['MobileCLIP', 'MobileCLIP2']:
            self.tokenizer = get_tokenizer('-'.join([clip_type, vit_type]))
            self.net = self.reparameterize_model(self.net)
            # self.fuse = FuseLite(
            #     in_channels=[64, 128, 256, 512], out_dim=512, target_size=(14, 14)).half()  # 没有权重,后续增强模块,另外加一个 fuse 权重
        else:
            self.tokenizer = tokenizer.tokenize
        # self.net = self.net.half()
        self.clip_type = clip_type
        self.vit_type = vit_type
        self.model_type = model_type
        self.feature_up = feature_up
        self.cls_token_lambda = cls_token_lambda
        self.output_cls_token = cls_token_lambda != 0
        self.bg_idx = bg_idx

        if self.clip_type == 'BLIP':
            self.patch_size = self.net.visual_encoder.patch_size
        elif clip_type in ['MobileCLIP', 'MobileCLIP2']:
            self.patch_size = 32
        else:
            self.patch_size = self.net.visual.patch_size
        if isinstance(self.patch_size, int):
            self.patch_size = (self.patch_size, self.patch_size)
        query_words, self.query_idx = get_cls_idx(name_path)
        self.qw = query_words
        self.num_queries = len(query_words)
        self.num_classes = max(self.query_idx) + 1
        self.query_idx = torch.Tensor(
            self.query_idx).to(torch.int64).to(device)

        query_features = []

        with torch.no_grad():  # sub_imagenet_template, openai_imagenet_template
            for qw in query_words:
                if self.clip_type == 'BLIP':
                    query = self.net.tokenizer([temp(qw) for temp in openai_imagenet_template], padding='max_length',
                                               truncation=True, max_length=35,
                                               return_tensors="pt").to(device)
                    text_output = self.net.text_encoder(query.input_ids, attention_mask=query.attention_mask,
                                                        mode='text')
                    feature = F.normalize(self.net.text_proj(
                        text_output.last_hidden_state[:, 0, :]))
                else:
                    query = self.tokenizer(
                        [temp(qw) for temp in openai_imagenet_template]).to(device)
                    feature = self.net.encode_text(query)
                    feature /= feature.norm(dim=-1, keepdim=True)
                feature = feature.mean(dim=0)
                feature /= feature.norm()
                query_features.append(feature.unsqueeze(0))
        self.query_features = torch.cat(query_features, dim=0)

        self.dtype = self.query_features.dtype
        self.ignore_residual = ignore_residual
        self.logit_scale = logit_scale
        self.prob_thd = prob_thd
        self.slide_stride = slide_stride
        self.slide_crop = slide_crop

        if feature_up:
            self.feat_dim = self.query_features.shape[-1]
            self.upsampler = get_upsampler(
                feature_up_cfg['model_name'], self.feat_dim).cuda().half()
            if (feature_up_cfg['model_path'] != None):
                ckpt = torch.load(feature_up_cfg['model_path'])['state_dict']
                weights_dict = {k[10:]: v for k, v in ckpt.items()}
                self.upsampler.load_state_dict(weights_dict, strict=True)

    def forward_feature(self, img, logit_size=None):
        if type(img) == list:
            img = img[0]
        if self.clip_type == 'BLIP':
            img = F.interpolate(img, size=(
                self.slide_crop, self.slide_crop), mode='bilinear', align_corners=False)  # 预处理单拉出来
            image_features = self.net.visual_encoder(img, self.ignore_residual)
            image_features = self.net.vision_proj(image_features[:, 1:, ])
        elif self.model_type == 'GEM':
            image_features = self.net.visual(img)
        elif self.clip_type in ['MobileCLIP', 'MobileCLIP2']:
            # 检查输入类型并相应处理,临时代码，这里有降低代码速度的问题
            if isinstance(img, torch.Tensor):
                # 如果是张量，说明已经经过数据预处理，需要转换回 PIL 或直接使用
                if img.dim() == 4 and img.shape[0] == 1:
                    # 单张图片的 batch 维度，移除它
                    img_tensor = img.squeeze(0)
                else:
                    img_tensor = img
                # 检查是否需要转换为 PIL
                if hasattr(img_tensor, 'shape') and len(img_tensor.shape) == 3:
                    # 转换张量为 PIL 图像
                    # 假设输入是 CHW 格式，值范围 0-255
                    if img_tensor.dtype == torch.float32 and img_tensor.max() <= 1.0:
                        # 如果是 0-1 范围的浮点数，转换为 0-255
                        img_tensor = (img_tensor * 255).byte()
                    elif img_tensor.dtype != torch.uint8:
                        img_tensor = img_tensor.byte()
                    # 在 permute 操作之前添加通道反转
                    img_rgb = img_tensor[[2, 1, 0], :, :]  # BGR to RGB
                    img_pil = Image.fromarray(
                        img_rgb.permute(1, 2, 0).cpu().numpy())
                img = img_pil
            img = self.preprocess(img)
            img = img.unsqueeze(0).to(self.device).half()
            temp_features = self.net.visual.forward(img, self.output_cls_token)
            if self.output_cls_token:
                cls_token, out_features,final_feature = temp_features
                # out_feature = self.zero_fuse_weighted(out_features)
                # image_features = [cls_token,out_feature]
                # image_features = [cls_token, out_features[-1]]
                image_features = [cls_token, final_feature]

            else:
                out_features = temp_features
                image_features = out_features[-1]
        else:
            image_features = self.net.encode_image(
                img, self.model_type, self.ignore_residual, self.output_cls_token)
        if self.output_cls_token:
            image_cls_token, image_features = image_features
            image_cls_token /= image_cls_token.norm(dim=-1, keepdim=True)
            cls_logits = image_cls_token @ self.query_features.T

        # featup
        if self.feature_up:
            feature_w, feature_h = img[0].shape[-2] // self.patch_size[0], img[0].shape[-1] // self.patch_size[1]
            image_w, image_h = img[0].shape[-2], img[0].shape[-1]
            # if not self.clip_type in ['MobileCLIP', 'MobileCLIP2']:
            image_features = image_features.permute(0, 2, 1).view(
                1, self.feat_dim, feature_w, feature_h)
            with torch.cuda.amp.autocast():
                image_features = self.upsampler(image_features, img).half()
            image_features = image_features.view(
                1, self.feat_dim, image_w * image_h).permute(0, 2, 1)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        logits = image_features @ self.query_features.T
        if self.output_cls_token:
            logits = logits + cls_logits * self.cls_token_lambda
    
        if self.feature_up:
            w, h = img[0].shape[-2], img[0].shape[-1]
        else:
            w, h = img[0].shape[-2] // self.patch_size[0], img[0].shape[-1] // self.patch_size[1]
        out_dim = logits.shape[-1]
        logits = logits.permute(0, 2, 1).reshape(-1, out_dim, w, h)

        if logit_size == None:
            logits = nn.functional.interpolate(
                logits, size=img.shape[-2:], mode='bilinear')
        else:
            logits = nn.functional.interpolate(
                logits, size=logit_size, mode='bilinear')

        return logits

    def forward_slide(self, img, img_metas, stride=112, crop_size=224):
        """Inference by sliding-window with overlap.
        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """
        if type(img) == list:
            img = img[0].unsqueeze(0)  # 临时方案：只处理第一张图片,针对自定义 preprocess
        if type(stride) == int:
            stride = (stride, stride)
        if type(crop_size) == int:
            crop_size = (crop_size, crop_size)

        h_stride, w_stride = stride
        h_crop, w_crop = crop_size
        batch_size, _, h_img, w_img = img.shape
        out_channels = self.num_queries
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, out_channels, h_img, w_img)).half()
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img)).half()
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]

                # pad image when (image_size % patch_size != 0)
                H, W = crop_img.shape[2:]
                pad = self.compute_padsize(H, W, self.patch_size[0])

                if any(pad):
                    crop_img = nn.functional.pad(crop_img, pad)

                crop_seg_logit = self.forward_feature(crop_img)

                # mask cutting for padded image
                if any(pad):
                    l, t = pad[0], pad[2]
                    crop_seg_logit = crop_seg_logit[:, :, t:t + H, l:l + W]

                preds += nn.functional.pad(crop_seg_logit,
                                           (int(x1), int(preds.shape[3] - x2), int(y1),
                                            int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0

        preds = preds / count_mat
        img_size = img_metas[0]['ori_shape'][:2]
        logits = nn.functional.interpolate(
            preds, size=img_size, mode='bilinear')

        return logits

    @torch.no_grad()
    def predict(self, inputs, data_samples):
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
        inputs = inputs.half()
        if self.slide_crop > 0:
            seg_logits = self.forward_slide(
                inputs, batch_img_metas, self.slide_stride, self.slide_crop)
        else:
            seg_logits = self.forward_feature(
                inputs, batch_img_metas[0]['ori_shape'])

        return self.postprocess_result(seg_logits, data_samples)
    # @torch.no_grad()
    # def predict(self, inputs, data_samples):
    #     if data_samples is not None:
    #         batch_img_metas = [
    #             data_sample.metainfo for data_sample in data_samples
    #         ]
    #     else:
    #         # 保持与原来完全一致的逻辑
    #         if isinstance(inputs, list):
    #             # 对于列表输入，使用第一个元素的形状信息
    #             sample_shape = inputs[0].shape if len(
    #                 inputs) > 0 else (1, 3, 224, 224)
    #             batch_size = len(inputs)
    #         else:
    #             sample_shape = inputs.shape
    #             batch_size = sample_shape[0]

    #         batch_img_metas = [
    #             dict(
    #                 ori_shape=sample_shape[2:],
    #                 img_shape=sample_shape[2:],
    #                 pad_shape=sample_shape[2:],
    #                 padding_size=[0, 0, 0, 0])
    #         ] * batch_size

    #     # 根据模型类型处理输入
    #     if self.clip_type in ['MobileCLIP', 'MobileCLIP2']:
    #         # MobileCLIP 保持列表格式，在后续处理中应用 preprocess
    #         processed_inputs = inputs
    #     else:
    #         # 其他模型转换为张量并转为 half 精度
    #         if isinstance(inputs, list):
    #             processed_inputs = torch.stack(inputs).half()  # 将列表转换为张量
    #         else:
    #             processed_inputs = inputs.half()

    #     if self.slide_crop > 0:
    #         seg_logits = self.forward_slide(
    #             processed_inputs, batch_img_metas, self.slide_stride, self.slide_crop)
    #     else:
    #         seg_logits = self.forward_feature(
    #             processed_inputs, batch_img_metas[0]['ori_shape'])

    #     return self.postprocess_result(seg_logits, data_samples)

    def postprocess_result(self, seg_logits, data_samples):
        batch_size = seg_logits.shape[0]
        for i in range(batch_size):
            seg_logits = seg_logits[i] * self.logit_scale
            seg_logits = seg_logits.softmax(0)  # n_queries * w * h

            num_cls, num_queries = max(self.query_idx) + 1, len(self.query_idx)
            if num_cls != num_queries:
                seg_logits = seg_logits.unsqueeze(0)
                cls_index = nn.functional.one_hot(self.query_idx)
                cls_index = cls_index.T.view(num_cls, num_queries, 1, 1)
                seg_logits = (seg_logits * cls_index).max(1)[0]

            seg_pred = seg_logits.argmax(0, keepdim=True)
            seg_pred[seg_logits.max(0, keepdim=True)[0]
                     < self.prob_thd] = self.bg_idx

            if data_samples is None:
                return seg_pred
            else:
                data_samples[i].set_data({
                    'seg_logits':
                        PixelData(**{'data': seg_logits}),
                    'pred_sem_seg':
                        PixelData(**{'data': seg_pred})
                })
        return data_samples

    def compute_padsize(self, H: int, W: int, patch_size: int):
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

    def zero_fuse_weighted(self, feats, weights=[0.1, 0.2, 0.3, 0.4], target_size=(14, 14)):
        outs = []
        weights = [w / sum(weights) for w in weights]

        # 找到最大通道数
        max_channels = max(f.shape[1] for f in feats)

        for f, w in zip(feats, weights):
            f_up = F.interpolate(f, size=target_size,
                                 mode='bilinear', align_corners=False)

            # 如果通道数不足，用零填充
            if f_up.shape[1] < max_channels:
                pad_size = max_channels - f_up.shape[1]
                padding = torch.zeros(
                    f_up.shape[0], pad_size, f_up.shape[2], f_up.shape[3]).to(f_up.device)
                f_up = torch.cat([f_up, padding], dim=1)

            outs.append(f_up * w)

        return sum(outs)

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

    def reparameterize_model(self, model: torch.nn.Module) -> nn.Module:
        """Method returns a model where a multi-branched structure
            used in training is re-parameterized into a single branch
            for inference.

        Args:
            model: MobileOne model in train mode.

        Returns:
            MobileOne model in inference mode.
        """
        # Avoid editing original graph
        model = copy.deepcopy(model)
        for module in model.modules():
            if hasattr(module, "reparameterize"):
                module.reparameterize()
        return model

    def cnn_module_fuse(self, img):
        """
        MobileCLIP中间特征信息,拿来在这里 fuse 之后再进入 Upsampler
        """
        if self.fuse is not None:
            image_features_dict = self.net.forward_intermediates(img)
            # 添加所有中间特征作为 fuse 的输入
            if self.output_cls_token:
                cls_token = image_features_dict['image_features']

            # 将所有中间层特征传递给 fuse 模块
            image_intermediates = image_features_dict['image_intermediates']
            fused_features = self.fuse(image_intermediates)

            return cls_token, fused_features

        return None


def get_cls_idx(path):
    with open(path, 'r') as f:
        name_sets = f.readlines()
    num_cls = len(name_sets)

    class_names, class_indices = [], []
    for idx in range(num_cls):
        names_i = name_sets[idx].split(',')
        class_names += names_i
        class_indices += [idx for _ in range(len(names_i))]
    class_names = [item.replace('\n', '') for item in class_names]
    return class_names, class_indices
