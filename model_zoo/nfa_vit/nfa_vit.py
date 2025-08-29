import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_

from IMDLBenCo.registry import MODELS

import sys
sys.path.append('.')

from .DnCNN import DnCNN
from .noisebackbone_segformer_b0 import MixVisionTransformer_b0
from .imagebackbone_segformer_b2 import MixVisionTransformer_b2
from .decoderhead import Multiple


@MODELS.register_module()
class NFA_ViT_modify1(nn.Module):
    def __init__(
        self, 
        np_pretrain_weights: str = None,
        seg_b0_pretrain_weights: str = None,
        seg_b2_pretrain_weights: str = None,
    ):
        super().__init__()
        self.noise_extractor = DnCNN(
            nplanes_in = 3,
            nplanes_out = 1,
            features = 64,
            kernel = 3,
            depth = 17,
            activation = 'relu',
            lastact = 'linear',
            residual = True,
            bn = True,            
        )
        
        self.noise_backbone = MixVisionTransformer_b0(in_chans = 1, sparse_ratio = 0.25)
        
        self.image_backbone = MixVisionTransformer_b2(in_chans = 3, sparse_rate = 2)
        
        self.cls_decoder = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 1)
        )
        
        self.seg_decoder = Multiple()
        # self.seg_decoder = SegFormerHead(num_classes=1, in_channels=[64, 128, 320, 512], embedding_dim=768, dropout_ratio=0.1)
        
        
        self.seg_loss = nn.BCEWithLogitsLoss()
        self.cls_loss = nn.BCEWithLogitsLoss()
        
        self.apply(self._init_weights)
        
        ######################################################### np Load
        assert np_pretrain_weights is not None, "np_pretrain_weights is required"
        self.noise_extractor.layers.load_state_dict(torch.load(np_pretrain_weights), strict = True)
        ######################################################### 
        
        
        ########################################################## seg-b0 Load
        assert seg_b0_pretrain_weights is not None, "seg_b0_pretrain_weights is required"
        weight = torch.load(seg_b0_pretrain_weights)
        if 'patch_embed1.proj.weight' in weight:
            original_weight = weight['patch_embed1.proj.weight']  # [32, 3, 7, 7]
            original_bias = weight['patch_embed1.proj.bias']      # [32]
            
            # 将3通道权重转换为1通道权重（取平均值）
            new_weight = original_weight.mean(dim=1, keepdim=True)  # [32, 1, 7, 7]
            
            # 更新权重字典
            weight['patch_embed1.proj.weight'] = new_weight
            
        self.noise_backbone.load_state_dict(weight, strict = False)
        ########################################################## 
        
        
        ########################################################## seg-b2 Load
        assert seg_b2_pretrain_weights is not None, "seg_b2_pretrain_weights is required"
        weight = torch.load(seg_b2_pretrain_weights)
        self.image_backbone.load_state_dict(weight, strict = False)
        ##########################################################  

        
        self.noise_extractor.requires_grad_(False)
        
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def forward(self, image, mask, label, *args, **kwargs):
        noise = self.noise_extractor(image)
        
        noise_features, noise_guided_masks = self.noise_backbone(noise)
        image_features = self.image_backbone(image, noise_guided_masks)   
        
        
        pred_mask = self.seg_decoder(image_features) 
        # pred_mask = self.seg_decoder(image_features, noise_features)
        pred_mask = F.interpolate(pred_mask, size=(image.shape[2], image.shape[3]), mode='bilinear', align_corners=False)
        
        pred = self.cls_decoder(image_features[-1])
        
        seg_loss = self.seg_loss(pred_mask, mask)
        
        label = label.unsqueeze(-1).float()
        cls_loss = self.cls_loss(pred, label)
        
        loss = seg_loss + 0.5 * cls_loss
        # loss = seg_loss
        
        
        pred_mask = torch.sigmoid(pred_mask.float())
        
        pred_label = torch.sigmoid(pred.float()).squeeze()
        
        output_dict = {
            # loss for backward
            "backward_loss": loss,
            # predicted mask, will calculate for metrics automatically
            "pred_mask": pred_mask,
            # predicted binaray label, will calculate for metrics automatically
            "pred_label": pred_label,

            # ----values below is for visualization----
            # automatically visualize with the key-value pairs
            "visual_loss": {
                "predict_loss": loss,
                'predict_mask_loss': seg_loss,
                'predict_label_loss': cls_loss,
            },

            "visual_image": {
                "pred_mask": pred_mask,
            }
            # -----------------------------------------
        }
        return output_dict
        
