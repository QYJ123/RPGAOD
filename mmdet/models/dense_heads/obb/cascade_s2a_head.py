from __future__ import division

import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, normal_init, bias_init_with_prob
from mmdet.core import (get_bbox_dim, build_bbox_coder, build_assigner, build_sampler,    
                        build_anchor_generator, 
                        images_to_levels, multi_apply,force_fp32 )

from mmcv.ops import DeformConv2dPack, DeformConv2d
from mmdet.models.builder import HEADS, build_loss
from .obb_anchor_head import OBBAnchorHead


class AlignConv(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 deformable_groups=1):
        super(AlignConv, self).__init__()
        self.kernel_size = kernel_size
        self.deform_conv = DeformConv(in_channels,
                                      out_channels,
                                      kernel_size=kernel_size,
                                      padding=(kernel_size - 1) // 2,
                                      deformable_groups=deformable_groups)
        self.relu = nn.ReLU(inplace=True)

    def init_weights(self):
        normal_init(self.deform_conv, std=0.01)

    @torch.no_grad()
    def get_offset(self, anchors, featmap_size, stride):
        dtype, device = anchors.dtype, anchors.device
        feat_h, feat_w = featmap_size
        pad = (self.kernel_size - 1) // 2
        idx = torch.arange(-pad, pad + 1, dtype=dtype, device=device)
        yy, xx = torch.meshgrid(idx, idx)
        xx = xx.reshape(-1)
        yy = yy.reshape(-1)

        # get sampling locations of default conv
        xc = torch.arange(0, feat_w, device=device, dtype=dtype)
        yc = torch.arange(0, feat_h, device=device, dtype=dtype)
        yc, xc = torch.meshgrid(yc, xc)
        xc = xc.reshape(-1)
        yc = yc.reshape(-1)
        x_conv = xc[:, None] + xx
        y_conv = yc[:, None] + yy

        # get sampling locations of anchors
        x_ctr, y_ctr, w, h, a = torch.unbind(anchors, dim=1)
        x_ctr, y_ctr, w, h = x_ctr / stride, y_ctr / stride, w / stride, h / stride
        cos, sin = torch.cos(a), torch.sin(a)
        dw, dh = w / self.kernel_size, h / self.kernel_size
        x, y = dw[:, None] * xx, dh[:, None] * yy
        xr = cos[:, None] * x - sin[:, None] * y
        yr = sin[:, None] * x + cos[:, None] * y
        x_anchor, y_anchor = xr + x_ctr[:, None], yr + y_ctr[:, None]
        # get offset filed
        offset_x = x_anchor - x_conv
        offset_y = y_anchor - y_conv
        # x, y in anchors is opposite in image coordinates,
        # so we stack them with y, x other than x, y
        offset = torch.stack([offset_y, offset_x], dim=-1)
        # NA,ks*ks*2
        offset = offset.reshape(anchors.size(
            0), -1).permute(1, 0).reshape(-1, feat_h, feat_w)
        return offset

    def forward(self, x, anchors, stride):
        num_imgs, _, H, W = x.size()
        offset_list = [
            self.get_offset(anchors[i], (H, W), stride)
            for i in range(num_imgs)
        ]
        offset_tensor = torch.stack(offset_list, dim=0)
        x = self.relu(self.deform_conv(x, offset_tensor))
        return x


@HEADS.register_module()
class CascadeS2ANetHead(OBBAnchorHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 stacked_convs=2,
                 with_align=True,
                 bbox_type='obb',
                 reg_dim=None,
                 anchor_generator=None,
                 bbox_coder=dict(
                     type='DeltaXYWHBBoxCoder',
                     target_means=(.0, .0, .0, .0),
                     target_stds=(1.0, 1.0, 1.0, 1.0)),
                 reg_decoded_bbox=False,
                 background_label=None,
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_bbox=dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
                 conv_cfg=None,
                 norm_cfg=None,
                 train_cfg=None,
                 test_cfg=None):
        super(CascadeS2ANetHead, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
	self.with_align = with_align
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.feat_channels = feat_channels
        self.bbox_type = bbox_type
        self.reg_dim = get_bbox_dim(self.bbox_type) \
            if reg_dim is None else reg_dim
        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        # TODO better way to determine whether sample or not
        self.sampling = loss_cls['type'] not in [
            'FocalLoss', 'GHMC', 'QualityFocalLoss'
        ]
        if self.use_sigmoid_cls:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1

        if self.cls_out_channels <= 0:
            raise ValueError(f'num_classes={num_classes} is too small')
        self.reg_decoded_bbox = reg_decoded_bbox

        self.background_label = (
            num_classes if background_label is None else background_label)
        # background_label should be either 0 or num_classes
        assert (self.background_label == 0
                or self.background_label == num_classes)

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            # use PseudoSampler when sampling is False
            if self.sampling and hasattr(self.train_cfg, 'sampler'):
                sampler_cfg = self.train_cfg.sampler
            else:
                sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
        self.fp16_enabled = False

        

        if anchor_generator is None:
            # Will Pass By S2AHead
            self.anchor_generator = None
            self.num_anchors = 1
            self.with_prior = True
        else:
            self.anchor_generator = build_anchor_generator(anchor_generator)
            # usually the numbers of anchors for each level are the same
            # except SSD detectors
            self.num_anchors = self.anchor_generator.num_base_anchors[0]
            self.with_prior = False
        self._init_layers()

    def _init_layers(self):
        """Initialize layers of the head."""
        if self.with_align:
            self.align_conv = AlignConv(
                self.feat_channels, self.feat_channels, 3)

	self.relu = nn.ReLU(inplace=True)

        self.reg_convs = nn.ModuleList()
        self.cls_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.feat_channels
            self.reg_convs.append(
                ConvModule(
                    self.feat_channels,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1))
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1))

        self.odm_cls = nn.Conv2d(
            self.feat_channels, self.cls_out_channels, 3, padding=1)
        self.odm_reg = nn.Conv2d(self.feat_channels, 5, 3, padding=1)

    def init_weights(self):
        """Initialize weights of the head."""
        if self.with_align:
            self.align_conv.init_weights()
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.odm_cls, std=0.01, bias=bias_cls)
        normal_init(self.odm_reg, std=0.01)

    def forward_single(self, x, anchors, stride):
        # feature alignment
        if self.with_align:
            aligned_feat = self.align_conv(x, anchors, stride)
        else:
            aligned_feat = x

        reg_feat = aligned_feat
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        bbox_pred = self.bbox_reg(reg_feat)

        cls_feat = aligned_feat
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        cls_score = self.bbox_cls(cls_feat)

        return cls_score, bbox_pred

    def forward(self, feats, anchor_list):
        """
        The forward function should do two things:
        1. anchor refinement by decoding the regressed box
        2. feature alignment by alignment convolution
        So it takes feats and anchors as input,
        and outputs cls_score, bbox_pred and refined anchors
        :param feats:
        :param anchors:
        :return:
        """
        # Transform img level to feature level
        num_imgs = len(anchor_list)
        num_levels = len(anchor_list[0])
        mlvl_anchor_list = [
            [anchor_list[j][i] for j in range(num_imgs)]
            for i in range(num_levels)
        ]

        return multi_apply(self.forward_single, feats, mlvl_anchor_list,\   
                           self.anchor_generator.strides)

    def get_prior_anchors(self,
                          featmap_sizes,
                          refine_anchors,
                          img_metas,
                          is_train=True,
                          device='cuda'):
        num_levels = len(featmap_sizes)

        refine_anchors_list = []
        for img_id, img_meta in enumerate(img_metas):
            mlvl_refine_anchors = []
            for i in range(num_levels):
                refine_anchor = refine_anchors[i][img_id].reshape(-1, 5)
                mlvl_refine_anchors.append(refine_anchor)
            refine_anchors_list.append(mlvl_refine_anchors)

        valid_flag_list = []
        if is_train:
            for img_id, img_meta in enumerate(img_metas):
                multi_level_flags = self.anchor_generator.valid_flags(
                    featmap_sizes, img_meta['pad_shape'], device)
                valid_flag_list.append(multi_level_flags)

        return refine_anchors_list, valid_flag_list

    def loss(self,
             cls_scores,
             bbox_preds,
             gt_obboxes,
             gt_labels,
             prior_anchors,
             img_metas,
             gt_bboxes_ignore=None):
        if prior_anchors is None:
            assert not self.with_prior
            return super().loss(cls_scores, bbox_preds, gt_obboxes, gt_labels, img_metas, gt_bboxes_ignore)
        else:
            assert self.with_prior
            featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
            assert len(featmap_sizes) == self.anchor_generator.num_levels

            device = cls_scores[0].device

            anchor_list, valid_flag_list = self.get_prior_anchors(
                featmap_sizes, prior_anchors, img_metas, device=device)
            label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
            cls_reg_targets = self.get_targets(
                anchor_list,
                valid_flag_list,
                gt_obboxes,
                img_metas,
                gt_bboxes_ignore_list=gt_bboxes_ignore,
                gt_labels_list=gt_labels,
                label_channels=label_channels)
            if cls_reg_targets is None:
                return None
            (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
             num_total_pos, num_total_neg) = cls_reg_targets
            num_total_samples = (
                num_total_pos + num_total_neg if self.sampling else num_total_pos)

            # anchor number of multi levels
            num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
            # concat all level anchors and flags to a single tensor
            concat_anchor_list = []
            for i in range(len(anchor_list)):
                concat_anchor_list.append(torch.cat(anchor_list[i]))
            all_anchor_list = images_to_levels(concat_anchor_list,
                                               num_level_anchors)

            losses_cls, losses_bbox = multi_apply(
                self.loss_single,
                cls_scores,
                bbox_preds,
                all_anchor_list,
                labels_list,
                label_weights_list,
                bbox_targets_list,
                bbox_weights_list,
                num_total_samples=num_total_samples)
            return dict(loss_cls=losses_cls, loss_bbox=losses_bbox)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   prior_anchors,
                   img_metas,
                   cfg=None,
                   rescale=False):
        if prior_anchors is None:
            assert not self.with_prior
            return super(CascadeS2ANetHead, self).get_bboxes(cls_scores, bbox_preds, img_metas, cfg, rescale)
        else:
            assert self.with_prior
            assert len(cls_scores) == len(bbox_preds)
            num_levels = len(cls_scores)

            device = cls_scores[0].device
            featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
            refine_anchors = self.get_prior_anchors(
                featmap_sizes, prior_anchors, img_metas, is_train=False, device=device)
            mlvl_anchors = refine_anchors[0][0]

            result_list = []
            for img_id in range(len(img_metas)):
                cls_score_list = [
                    cls_scores[i][img_id].detach() for i in range(num_levels)
                ]
                bbox_pred_list = [
                    bbox_preds[i][img_id].detach() for i in range(num_levels)
                ]
                img_shape = img_metas[img_id]['img_shape']
                scale_factor = img_metas[img_id]['scale_factor']
                proposals = self._get_bboxes_single(cls_score_list, bbox_pred_list,
                                                    mlvl_anchors, img_shape,
                                                    scale_factor, cfg, rescale)
                result_list.append(proposals)
            return result_list
