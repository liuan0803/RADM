# ========================================
# Modified by Fengheng Li
# ========================================
# Modified by Shoufa Chen
# ========================================
# Modified by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
RADM Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
import math

import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F

from detectron2.modeling.poolers import ROIPooler
from detectron2.structures import Boxes
_DEFAULT_SCALE_CLAMP = math.log(100000.0 / 16)

class VisualTextualRelationAwareModule(nn.Module):
    '''
    reference: LAVT: Language-Aware Vision Transformer for Referring Image Segmentation
    '''
    def __init__(self, dim, v_in_channels, l_in_channels, key_channels, value_channels, propose_num, num_heads=2, dropout=0.0):
        super(VisualTextualRelationAwareModule, self).__init__()
        
        self.vis_project = nn.Sequential(nn.Conv1d(dim, dim, 1, 1),  # the init function sets bias to 0 if bias is True
                                         nn.GELU(),
                                         nn.Dropout(dropout)
                                        )
               
        self.image_lang_att = VisualTextualAtten(v_in_channels,  # v_in
                                                            l_in_channels,  # l_in
                                                            key_channels,  # key
                                                            value_channels,  # value
                                                            propose_num=propose_num,
                                                            out_channels=value_channels,  # out
                                                            num_heads=num_heads)
        
        self.project_mm = nn.Sequential(nn.Conv1d(value_channels, value_channels, 1, 1),
                                        nn.GELU(),
                                        nn.Dropout(dropout)
                                        )

    def forward(self, x, l, l_mask, PE):
        # input x shape: (B, H*W, dim)
        vis = self.vis_project(x.permute(0, 2, 1))  # (B, dim, H*W)
        
        lang = self.image_lang_att(x, l, l_mask, PE)  # (B, H*W, dim)
        
        lang = lang.permute(0, 2, 1)  # (B, dim, H*W)

        mm = torch.mul(vis, lang)
        mm = self.project_mm(mm)  # (B, dim, H*W)

        mm = mm.permute(0, 2, 1)  # (B, H*W, dim)

        return mm


class VisualTextualAtten(nn.Module):
    def __init__(self, v_in_channels, l_in_channels, key_channels, value_channels, propose_num, out_channels=None, num_heads=1):
        super(VisualTextualAtten, self).__init__()
        '''
        x shape: (B, H*W, v_in_channels)
        l input shape: (B, l_in_channels, N_l)
        l_mask shape: (B, N_l, 1)
        '''
        self.propose_num = propose_num
        self.v_in_channels = v_in_channels
        self.l_in_channels = l_in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        self.num_heads = num_heads
        if out_channels is None:
            self.out_channels = self.value_channels
        self.linear = nn.Linear(50, 49)
        # Keys: textual features: (B, l_in_channels, #words)
        # avoid any form of spatial normalization because a sentence contains many padding 0s
        self.f_key = nn.Sequential(
            nn.Conv1d(self.l_in_channels, self.key_channels, kernel_size=1, stride=1),
        )

        # Queries: visual features: (B, p, H*W, v_in_channels)
        self.f_query = nn.Sequential(
            nn.Conv1d(self.v_in_channels, self.key_channels, kernel_size=1, stride=1),
            nn.InstanceNorm1d(self.key_channels),
        )

        # Values: textual features: (B, l_in_channels, #words)
        self.f_value = nn.Sequential(
            nn.Conv1d(self.l_in_channels, self.value_channels, kernel_size=1, stride=1),
        )

        # Out projection
        self.W = nn.Sequential(
            nn.Conv1d(self.value_channels, self.out_channels, kernel_size=1, stride=1),
            nn.InstanceNorm1d(self.out_channels),
        )

    def forward(self, x, l, l_mask, PE):
        # x shape: (B, p，H*W, v_in_channels)
        # l input shape: (B, l_in_channels, N_l)
        # l_mask shape: (B, N_l, 1)
        # PE shape :(B, p, key_channels)
        
        l = l.repeat(self.propose_num, 1, 1)
        # print(l_mask)
        l_mask = l_mask.repeat(self.propose_num, 1, 1)
        # print('after', l_mask)
        # print(x.shape)
        # print(l.shape)
        # print(PE.shape)
        B, HW = x.size(0), x.size(1)
        x = x.permute(0, 2, 1)  # (B, key_channels, H*W)
        l_mask = l_mask.permute(0, 2, 1)  # (B, N_l, 1) -> (B, 1, N_l)

        query = self.f_query(x)  # (B, key_channels, H*W) if Conv1D
        query = query.reshape(B, self.key_channels, -1)
        PE = PE.view(B, self.key_channels).unsqueeze(2)
        # print(PE.shape)
        # print(query.shape)
        query = torch.cat((query, PE), dim = 2)
        query = self.linear(query)    #(B, key_channel, 200, 49)
        query = query.reshape(B, self.key_channels, -1)
        query = query.permute(0, 2, 1)  # (B, H*W, key_channels)
        
        key = self.f_key(l)  # (B, key_channels, N_l)
        value = self.f_value(l)  # (B, self.value_channels, N_l)
        key = key * l_mask  # (B, key_channels, N_l)
        value = value * l_mask  # (B, self.value_channels, N_l)
        n_l = value.size(-1)
        query = query.reshape(B, HW, self.num_heads, self.key_channels//self.num_heads).permute(0, 2, 1, 3)
        # (b, num_heads, H*W, self.key_channels//self.num_heads)
        key = key.reshape(B, self.num_heads, self.key_channels//self.num_heads, n_l)
        # (b, num_heads, self.key_channels//self.num_heads, n_l)
        value = value.reshape(B, self.num_heads, self.value_channels//self.num_heads, n_l)
        # # (b, num_heads, self.value_channels//self.num_heads, n_l)
        l_mask = l_mask.unsqueeze(1)  # (b, 1, 1, n_l)

        sim_map = torch.matmul(query, key)  # (B, self.num_heads, H*W, N_l)
        sim_map = (self.key_channels ** -.5) * sim_map  # scaled dot product

        sim_map = sim_map + (1e4*l_mask - 1e4)  # assign a very small number to padding positions
        sim_map = F.softmax(sim_map, dim=-1)  # (B, num_heads, h*w, N_l)
        out = torch.matmul(sim_map, value.permute(0, 1, 3, 2))  # (B, num_heads, H*W, self.value_channels//num_heads)
        out = out.permute(0, 2, 1, 3).contiguous().reshape(B, HW, self.value_channels)  # (B, H*W, value_channels)
        out = out.permute(0, 2, 1)  # (B, value_channels, HW)
        out = self.W(out)  # (B, value_channels, HW)
        out = out.permute(0, 2, 1)  # (B, HW, value_channels)
        

        return out
    

class GeometryRelationAwareModule(nn.Module):
    def __init__(self,
                 topo_in_dim,
                 topo_out_dim=256,
                 drop_rate=0.,
                 embd_dim=64,
                 wave_length=1000,
                 fc_out_channels=1):
        super(GeometryRelationAwareModule, self).__init__()
        # used in calculate "weight_geo"
        self.linear = nn.Linear(embd_dim, fc_out_channels)
        self.relu = nn.ReLU(inplace=True)

        # W^v
        self.linear2 = nn.Linear(topo_in_dim, topo_out_dim)

        self.out_dim = embd_dim
        self.wave_length = wave_length
        self.topo_out_dim = topo_out_dim
        self.drop_rate = drop_rate

        self.init_weight()

    def init_weight(self):
        nn.init.normal_(self.linear.weight, 0, 0.01)
        nn.init.constant_(self.linear.bias, 0)

        nn.init.normal_(self.linear2.weight, 0, 0.01)
        nn.init.constant_(self.linear2.bias, 0)

    def build_relative_geo(self, rois, gt):
        assert rois.shape[1] == gt.shape[1]

        rois_repeat = rois[..., None].repeat(1, 1, gt.shape[0])  # broadcast
        gt_x = gt[:, 0]
        gt_y = gt[:, 1]
        gt_w = gt[:, 2]
        gt_h = gt[:, 3]
        gt_w = gt_w.maximum(torch.tensor(1e-3).to(gt.device))
        gt_h = gt_h.maximum(torch.tensor(1e-3).to(gt.device))

        # x
        rois_x = rois_repeat[:, 0, :]  # [512, gt.shape[0]]  ,broadcast
        rel_x = torch.abs(gt_x - rois_x) / gt_w
        rel_x = rel_x.maximum(torch.tensor(1e-3).to(rois.device))

        # y
        rois_y = rois_repeat[:, 1, :]
        rel_y = torch.abs(gt_y - rois_y) / gt_h
        rel_y = rel_y.maximum(torch.tensor(1e-3).to(rois.device))

        # w
        rois_w = rois_repeat[:, 2, :]
        rel_w = rois_w / gt_w
        rel_w = rel_w.maximum(torch.tensor(1e-3).to(rois.device))

        # h
        rois_h = rois_repeat[:, 3, :]
        rel_h = rois_h / gt_h
        rel_h = rel_h.maximum(torch.tensor(1e-3).to(rois.device))

        relative_geo = torch.stack([rel_x, rel_y, rel_w, rel_h], dim=-1).float()

        return torch.log(relative_geo)

    def extract_position_embedding(self, relative_geo, feat_dim=64, wave_length=1000):
        '''
        relative_geo: [num_rois, num_gt_rois, 4]
        reference:https://github.com/msracver/Relation-Networks-for-Object-Detection/
        /relation_rcnn/symbols/resnet_v1_101_rcnn_attention_1024_pairwise_position_multi_head_16_learn_nms.py
        '''
        feat_range = torch.arange(0, feat_dim / 8)
        dim_mat = torch.pow(torch.full((1,), wave_length), (8. / feat_dim) * feat_range).to(relative_geo.device)  # shape [1,8]
        dim_mat = dim_mat.reshape((1, 1, 1, -1))  # shape [1,1,1,8]

        relative_geo = torch.unsqueeze(100.0 * relative_geo, dim=-1)  # [num_rois, num_gt_rois, 4, 1]
        div_mat = relative_geo / dim_mat
        sin_mat = div_mat.sin()
        cos_mat = div_mat.cos()
        embedding = torch.stack([sin_mat, cos_mat], dim=-1)  # [num_rois, num_gt_rois, 4, feat_dim/4]
        embedding = embedding.flatten(2)  # [num_rois, num_gt_rois, feat_dim]

        return embedding
    
    def forward(self, rois, gt_rois, gt_bbox_feats, batch_num):
        """ 
        extract topology features for text tracking
        :param rois: shape (n, 5), [batch_ind, x1, y1, x2, y2] [batch_size, proposal_num, 4]
        :param gt_rois: ground truth (m, 5)   
        :param gt_bbox_feats: [m, channel, width, height] 
        :param batch_num: batch size
        :return: topology_feats [n, self.out_dim]
        """
        
        
        n = rois.shape[1]
        rois = rois.reshape(-1,4) #[n,4]
        rois = torch.cat((torch.tensor([[i for j in range(n)]for i in range(batch_num)]).reshape(-1, 1).to(rois.device), rois), dim=1) #[n,5]
        gt_rois = rois
        rois_xywh = torch.stack((rois[:, 0], (rois[:, 1] + rois[:, 3]) / 2, (rois[:, 2] + rois[:, 4]) / 2,
                                 rois[:, 3] - rois[:, 1], rois[:, 4] - rois[:, 2]), 1)
        
        gt_xywh = torch.stack((gt_rois[:, 0], (gt_rois[:, 1] + gt_rois[:, 3]) / 2, (gt_rois[:, 2] + gt_rois[:, 4]) / 2,
                               gt_rois[:, 3] - gt_rois[:, 1], gt_rois[:, 4] - gt_rois[:, 2]), 1)
        gt_bbox_feats_trans = self.linear2(gt_bbox_feats.view(gt_bbox_feats.size(0), -1))  # W^v
        topology_feats = torch.zeros((rois.shape[0], self.topo_out_dim)).to(rois.device) 
        
        for i in range(batch_num):
            rois_i = rois_xywh[rois[:, 0] == i, 1::]  # [n, 4]
    
            gt_i = gt_xywh[gt_rois[:, 0] == i, 1::]   # [m, 4]
            relative_geo = self.build_relative_geo(rois_i, gt_i)

            weight_geo = self.extract_position_embedding(relative_geo, self.out_dim, self.wave_length)  # [num_rois, num_gt_rois, out_dim]
            weight_geo = self.relu(self.linear(weight_geo)).squeeze(-1)  # [num_rois, num_gt_rois]
            weight_geo = nn.functional.softmax(weight_geo, dim=-1)
            weight_geo = nn.functional.dropout(weight_geo, p=self.drop_rate, training=self.training)
        
            topology_feats[rois[:, 0] == i] = torch.mm(weight_geo, gt_bbox_feats_trans[gt_rois[:, 0] == i])

        return topology_feats

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""

    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Dense(nn.Module):
    """A fully connected layer that reshapes outputs to feature maps."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.dense(x)
    
    
class DynamicHead(nn.Module):

    def __init__(self, cfg, roi_input_shape):
        super().__init__()

        # Build RoI.
        box_pooler = self._init_box_pooler(cfg, roi_input_shape)
        self.box_pooler = box_pooler

        # Build heads.
        num_classes = cfg.MODEL.RADM.NUM_CLASSES
        d_model = cfg.MODEL.RADM.HIDDEN_DIM
        dim_feedforward = cfg.MODEL.RADM.DIM_FEEDFORWARD
        nhead = cfg.MODEL.RADM.NHEADS
        dropout = cfg.MODEL.RADM.DROPOUT
        activation = cfg.MODEL.RADM.ACTIVATION
        num_heads = cfg.MODEL.RADM.NUM_HEADS
        rcnn_head = RCNNHead(cfg, d_model, num_classes, 64, dim_feedforward, nhead, dropout, activation)
        self.head_series = _get_clones(rcnn_head, num_heads)
        self.num_heads = num_heads
        self.return_intermediate = cfg.MODEL.RADM.DEEP_SUPERVISION
        
        # Gaussian random feature embedding layer for time
        self.d_model = d_model
        time_dim = d_model * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(d_model),
            nn.Linear(d_model, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # Init parameters.
        self.use_focal = cfg.MODEL.RADM.USE_FOCAL
        self.use_fed_loss = cfg.MODEL.RADM.USE_FED_LOSS
        self.num_classes = num_classes
        if self.use_focal or self.use_fed_loss:
            prior_prob = cfg.MODEL.RADM.PRIOR_PROB
            self.bias_value = -math.log((1 - prior_prob) / prior_prob)
        self._reset_parameters()

    def _reset_parameters(self):
        # init all parameters.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

            # initialize the bias for focal loss and fed loss.
            if self.use_focal or self.use_fed_loss:
                if p.shape[-1] == self.num_classes or p.shape[-1] == self.num_classes + 1:
                    nn.init.constant_(p, self.bias_value)

    @staticmethod
    def _init_box_pooler(cfg, input_shape):

        in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        return box_pooler

    def forward(self, features, init_bboxes, norm_bboxes, txt_features, txt_mask, t, init_features):
        # assert t shape (batch_size)
        time = self.time_mlp(t)

        inter_class_logits = []
        inter_pred_bboxes = []

        bs = len(features[0])
        bboxes = init_bboxes
        num_boxes = bboxes.shape[1]

        if init_features is not None:
            init_features = init_features[None].repeat(1, bs, 1)
            proposal_features = init_features.clone()
        else:
            proposal_features = None
        
        for head_idx, rcnn_head in enumerate(self.head_series):
            class_logits, pred_bboxes, proposal_features = rcnn_head(features, bboxes, norm_bboxes, txt_features, txt_mask, proposal_features, self.box_pooler, time)
            
            if self.return_intermediate:
                inter_class_logits.append(class_logits)
                inter_pred_bboxes.append(pred_bboxes)
            bboxes = pred_bboxes.detach()

        if self.return_intermediate:
            return torch.stack(inter_class_logits), torch.stack(inter_pred_bboxes)

        return class_logits[None], pred_bboxes[None]


class RCNNHead(nn.Module):

    def __init__(self, cfg, d_model, num_classes, topo_out_dim, dim_feedforward=2048, nhead=8, dropout=0.1, activation="relu",
                 scale_clamp: float = _DEFAULT_SCALE_CLAMP, bbox_weights=(2.0, 2.0, 1.0, 1.0)):
        super().__init__()

        self.propose_num = cfg.MODEL.RADM.NUM_PROPOSALS
        self.d_model = d_model
        self.withVTRAM = cfg.MODEL.RADM.withVTRAM
        self.withGRAM = cfg.MODEL.RADM.withGRAM
        self.d_fused = d_model
        # dynamic.
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.inst_interact = DynamicConv(cfg)
    
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        if self.withVTRAM:
            self.d_fused += d_model
            self.vis_text_att = VisualTextualRelationAwareModule(d_model, d_model, 768, d_model, d_model, self.propose_num, 2, 0)
            self.linear4 = nn.Linear(4, d_model)

        if self.withGRAM: 
            self.d_fused += topo_out_dim
            self.GRAM = GeometryRelationAwareModule(topo_in_dim=256*7*7, topo_out_dim=topo_out_dim)#256*7*7 RoIpooling output dimension
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
            
        
        self.activation = _get_activation_fn(activation)

        # block time mlp
        self.block_time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(d_model * 4, d_model * 2))

        # cls.
        num_cls = cfg.MODEL.RADM.NUM_CLS
        cls_module = list()
        for _ in range(num_cls):
            cls_module.append(nn.Linear(self.d_fused, self.d_fused, False))
            cls_module.append(nn.LayerNorm(self.d_fused))
            cls_module.append(nn.ReLU(inplace=True))
        self.cls_module = nn.ModuleList(cls_module)

        # reg.
        num_reg = cfg.MODEL.RADM.NUM_REG
        reg_module = list()
        for _ in range(num_reg):
            reg_module.append(nn.Linear(self.d_fused, self.d_fused, False))
            reg_module.append(nn.LayerNorm(self.d_fused))
            reg_module.append(nn.ReLU(inplace=True))
        self.reg_module = nn.ModuleList(reg_module)
        
        # pred.
        self.use_focal = cfg.MODEL.RADM.USE_FOCAL
        self.use_fed_loss = cfg.MODEL.RADM.USE_FED_LOSS
        if self.use_focal or self.use_fed_loss:
            self.class_logits = nn.Linear(self.d_fused, num_classes)
        else:
            self.class_logits = nn.Linear(self.d_fused, num_classes + 1)
        self.bboxes_delta = nn.Linear(self.d_fused, 4)
        self.scale_clamp = scale_clamp
        self.bbox_weights = bbox_weights

    def forward(self, features, bboxes, norm_bboxes, txt_features, txt_mask, pro_features, pooler,  time_emb):
        """
        :param bboxes: (N, nr_boxes, 4)
        :param norm_bboxes: (N, nr_boxes, 4) [0,1]
        :param pro_features: (N, nr_boxes, d_model)
        :param gt_rois: (m, 5)[batch_ind, x1,y1,x2,y2]
        """
        
        N, nr_boxes = bboxes.shape[:2]
        #print(N)
        # roi_feature.
        proposal_boxes = list()
        
        for b in range(N):
            proposal_boxes.append(Boxes(bboxes[b]))
        
        roi_features = pooler(features, proposal_boxes) 
        # the first head
        if pro_features is None:
            pro_features = roi_features.view(N, nr_boxes, self.d_model, -1).mean(-1)

        txt_features = txt_features.view(N, 768, -1)
        # GTRAM
        if self.withGRAM:
            top_features = self.GRAM(bboxes, bboxes, roi_features, N)  #[batch_size * num_proposals, 256]
        roi_features = roi_features.view(N * nr_boxes , self.d_model, -1).permute(0, 2, 1) # (N*p, 7*7, 256)
        
        #VTRAM
        if self.withVTRAM:
            PE = self.linear4(norm_bboxes)
            mm_fea = self.vis_text_att(roi_features, txt_features, txt_mask, PE)   #(B, p*w*h, 256)
            mm_fea = mm_fea.reshape(N * nr_boxes, self.d_model, -1).mean(-1)
       
        pro_features = pro_features.view(N, nr_boxes, self.d_model).permute(1, 0, 2)  #(B, p, 256)

        
        pro_features2 = self.self_attn(pro_features, pro_features, value=pro_features)[0]
        pro_features = pro_features + self.dropout1(pro_features2)
        pro_features = self.norm1(pro_features)
        roi_features = roi_features.permute(1, 0, 2) 
        # inst_interact.
        pro_features = pro_features.view(nr_boxes, N, self.d_model).permute(1, 0, 2).reshape(1, N * nr_boxes, self.d_model)
        # print(roi_features.shape) #(49, 800, 256)
        # print(pro_features.shape) #(1 ,800, 256)
        pro_features2 = self.inst_interact(pro_features, roi_features)
        #print(pro_features2.shape)#(800, 256)
        pro_features = pro_features + self.dropout2(pro_features2)
        obj_features = self.norm2(pro_features)
        #txt_features.shape #(1, 12, 768)
        
        
  
        # obj_feature.
        obj_features2 = self.linear2(self.dropout(self.activation(self.linear1(obj_features))))
        obj_features = obj_features + self.dropout3(obj_features2)
        obj_features = self.norm3(obj_features)
        
        fc_feature = obj_features.transpose(0, 1).reshape(N * nr_boxes, -1)
        
        scale_shift = self.block_time_mlp(time_emb)
        scale_shift = torch.repeat_interleave(scale_shift, nr_boxes, dim=0)
        scale, shift = scale_shift.chunk(2, dim=1)
        fc_feature = fc_feature * (scale + 1) + shift
      
        if self.withVTRAM:
            fc_feature = torch.cat((fc_feature, mm_fea), dim=1)       
        if self.withGRAM:
            fc_feature = torch.cat((top_features, fc_feature), dim=1)

        
        
        cls_feature = fc_feature.clone()
        reg_feature = fc_feature.clone()
        for cls_layer in self.cls_module:
            cls_feature = cls_layer(cls_feature)
        for reg_layer in self.reg_module:
            reg_feature = reg_layer(reg_feature)
        class_logits = self.class_logits(cls_feature)
        #reg_feature = torch.cat((reg_feature, class_logits), dim=1)
        bboxes_deltas = self.bboxes_delta(reg_feature)
        pred_bboxes = self.apply_deltas(bboxes_deltas, bboxes.view(-1, 4))
        
        return class_logits.view(N, nr_boxes, -1), pred_bboxes.view(N, nr_boxes, -1), obj_features

    def apply_deltas(self, deltas, boxes):
        """
        Apply transformation `deltas` (dx, dy, dw, dh) to `boxes`.

        Args:
            deltas (Tensor): transformation deltas of shape (N, k*4), where k >= 1.
                deltas[i] represents k potentially different class-specific
                box transformations for the single box boxes[i].
            boxes (Tensor): boxes to transform, of shape (N, 4)
        """
        boxes = boxes.to(deltas.dtype)

        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        wx, wy, ww, wh = self.bbox_weights
        dx = deltas[:, 0::4] / wx
        dy = deltas[:, 1::4] / wy
        dw = deltas[:, 2::4] / ww
        dh = deltas[:, 3::4] / wh

        # Prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max=self.scale_clamp)
        dh = torch.clamp(dh, max=self.scale_clamp)

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        pred_boxes = torch.zeros_like(deltas)
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w  # x1
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h  # y1
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w  # x2
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h  # y2

        return pred_boxes


class DynamicConv(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.hidden_dim = cfg.MODEL.RADM.HIDDEN_DIM
        self.dim_dynamic = cfg.MODEL.RADM.DIM_DYNAMIC
        self.num_dynamic = cfg.MODEL.RADM.NUM_DYNAMIC
        self.num_params = self.hidden_dim * self.dim_dynamic
        self.dynamic_layer = nn.Linear(self.hidden_dim, self.num_dynamic * self.num_params)

        self.norm1 = nn.LayerNorm(self.dim_dynamic)
        self.norm2 = nn.LayerNorm(self.hidden_dim)

        self.activation = nn.ReLU(inplace=True)

        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        num_output = self.hidden_dim * pooler_resolution ** 2
        self.out_layer = nn.Linear(num_output, self.hidden_dim)
        self.norm3 = nn.LayerNorm(self.hidden_dim)

    def forward(self, pro_features, roi_features):
        '''
        pro_features: (1,  N * nr_boxes, self.d_model)
        roi_features: (49, N * nr_boxes, self.d_model)
        '''
        features = roi_features.permute(1, 0, 2)
        parameters = self.dynamic_layer(pro_features).permute(1, 0, 2)
        #print(parameters.shape)
        param1 = parameters[:, :, :self.num_params].view(-1, self.hidden_dim, self.dim_dynamic)
        param2 = parameters[:, :, self.num_params:].view(-1, self.dim_dynamic, self.hidden_dim)
        #print(param1.shape)
        #print(param2.shape)
        #print(features.shape)
        features = torch.bmm(features, param1)
        features = self.norm1(features)
        features = self.activation(features)

        features = torch.bmm(features, param2)
        features = self.norm2(features)
        features = self.activation(features)

        features = features.flatten(1)
        features = self.out_layer(features)
        features = self.norm3(features)
        features = self.activation(features)

        return features


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
