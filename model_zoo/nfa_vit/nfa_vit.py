import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_

from IMDLBenCo.registry import MODELS

import sys
sys.path.append('.')

from DnCNN import DnCNN
from noisebackbone_segformer_b0 import MixVisionTransformer_b0
from imagebackbone_segformer_b2 import MixVisionTransformer_b2
from decoderhead import Multiple


@MODELS.register_module()
class NFA_ViT(nn.Module):
    def __init__(
        self, 
        np_pretrain_weights: str = None,
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
            nn.Linear(256, 1)
        )
        
        self.seg_decoder = Multiple()
        
        self.seg_loss = nn.BCEWithLogitsLoss()
        self.cls_loss = nn.BCEWithLogitsLoss()
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def forward(self, image, mask, label, *args, **kwargs):
        noise = self.noise_extractor(img)
        
        noise_feature, noise_guided_masks = self.noise_backbone(noise)
        
        
        image_features = self.image_backbone(image, noise_guided_masks)   
        
        pred_mask = self.seg_decoder(image_features)   
        pred_mask = F.interpolate(pred_mask, size=(image.shape[2], image.shape[3]), mode='bilinear', align_corners=False)
        
        pred = self.cls_decoder(image_features[-1])
        
        seg_loss = self.seg_loss(pred_mask, mask)
        cls_loss = self.cls_loss(pred, label)
        
        loss = seg_loss + cls_loss
        
        
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
        

        


if __name__ == '__main__':
    model = NFA_ViT()
    img = torch.randn(20, 3, 512, 512)
    mask = torch.zeros(20, 1, 512, 512)
    label = torch.zeros(20, 1)
    
    model(img, mask, label)