"""
Basic ceutrack model.
"""
import math
import os
from typing import List
import sys
import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones
from lib.utils.heapmap_utils import generate_heatmap
from lib.models.layers.head import build_box_head
from lib.models.ceutrack.vit import vit_base_patch16_224
from lib.models.ceutrack.vit_ce import vit_large_patch16_224_ce, vit_base_patch16_224_ce
# from lib.models.ceutrack.vit_ce_BACKUPS import vit_large_patch16_224_ce, vit_base_patch16_224_ce
from lib.utils.box_ops import box_xyxy_to_cxcywh
from torch.autograd import Variable
import numpy as np
from lib.train.actors.ceutrack import CEUTrackActor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
from lib.utils.box_ops import giou_loss
from torch.nn.functional import l1_loss
import matplotlib.pyplot as plt
from lib.utils.focal_loss import FocalLoss
import logging
import os
from torchvision.utils import make_grid
from attack.Noise_SGD import Att_SGD
from attack.Noise_SGD import keepGradUpdate
# logging.basicConfig(
#     filename='/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c01/DATA/chenqiang/AttackTracking/CEUTrack/output/test/tracking_results/log',
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     datefmt='%Y-%m-%d %H:%M:%S'
# )

class CEUTrack(nn.Module):
    """ This is the base class for ceutrack """

    def __init__(self, transformer, box_head, aux_loss=False, head_type="CORNER",initial_noise=None):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone = transformer
        self.box_head = box_head

        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)


        # # 初始化计数器
        # self.forward_counter = 0

        # self.noise = nn.Parameter(torch.zeros(1,3,256,256))
        self.unix = 0
        self.max_eps = 10 * self.unix 

        # Initialize self.noise from the external argument or as zero
        if initial_noise is None:
            self.noise = nn.Parameter(torch.zeros(1, 3, 256, 256))
        else:
            self.noise = nn.Parameter(initial_noise)

        self.optimizer = Att_SGD(
                [{"params": [self.noise], "lr": self.max_eps / 10, "momentum": 1, "sign": True}],
                max_eps=self.max_eps,
            )
       


    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                event_template: torch.Tensor,        # torch.Size([4, 1, 19, 10000])
                event_search: torch.Tensor,          # torch.Size([4, 1, 19, 10000])
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                ):

        # before feeding into backbone, we need to concat four vectors, or two two concat;
        x, aux_dict = self.backbone(z=template, x=search, event_z=event_template, event_x=event_search,
                                    ce_template_mask=ce_template_mask,
                                    ce_keep_rate=ce_keep_rate,
                                    return_last_attn=return_last_attn, )

        # Forward head
        feat_last = x
        if isinstance(x, list):
            feat_last = x[-1]
        out = self.forward_head(feat_last, None)

        out.update(aux_dict)
        out['backbone_feat'] = x
        return out
    
    def forward_attck_rgb_event_frame_sequce(self, template: torch.Tensor,
                search: torch.Tensor,
                event_template: torch.Tensor,        # torch.Size([4, 1, 19, 10000])
                event_search: torch.Tensor,          # torch.Size([4, 1, 19, 10000])
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                eps=10, 
                alpha=1, 
                iteration=10, 
                x_val_red_max=0, 
                x_val_red_min=1,
                x_val_green_max=0,
                x_val_green_min=1,
                x_val_blue_max=0,
                x_val_blue_min=1, 
                event_x_val_red_max=0,
                event_x_val_red_min=1,
                event_x_val_green_max=0,
                event_x_val_green_min=1,
                event_x_val_blue_max=0,
                event_x_val_blue_min=1,
                unit_pix = None,
                adv_bbox = None,
                att_per = 0,
                cfg = None,
                frame_id = 0,
                current_time = None,
                real_bbox= None,
                gt_gaussian_maps = None,
                fack_gaussian_maps = None,
                ):
            scale_factor = (4.7579 / 255)*2 
            scale_factor_new = 1
            #search = Variable(search.data,requires_grad=True) 
            ori_rgb_search = search
            rgb_adv_search = search + att_per * 1 * scale_factor_new # 初试为0
            rgb_adv_search[:,0,:,:] = torch.clamp(rgb_adv_search[:,0,:,:], max=x_val_red_max, min=x_val_red_min)
            rgb_adv_search[:,1,:,:] = torch.clamp(rgb_adv_search[:,1,:,:], max=x_val_green_max, min=x_val_green_min)
            rgb_adv_search[:,2,:,:] = torch.clamp(rgb_adv_search[:,2,:,:], max=x_val_blue_max, min=x_val_blue_min)
            rgb_adv_search = Variable(rgb_adv_search.data, requires_grad=True)
            rgb_adv_search.retain_grad()  # 保留 x_adv 的梯度

            ori_event_search = event_search
            event_adv_search = event_search + att_per * 1 * scale_factor_new # 初试为0
            event_adv_search[:,0,:,:] = torch.clamp(event_adv_search[:,0,:,:], max=event_x_val_red_max, min=event_x_val_red_min)
            event_adv_search[:,1,:,:] = torch.clamp(event_adv_search[:,1,:,:], max=event_x_val_green_max, min=event_x_val_green_min)
            event_adv_search[:,2,:,:] = torch.clamp(event_adv_search[:,2,:,:], max=event_x_val_blue_max, min=event_x_val_blue_min)
            event_adv_search = Variable(event_adv_search.data, requires_grad=True)
            event_adv_search.retain_grad()  # 保留 x_adv 的梯度
            fack_bbox = adv_bbox 
            fack_boxes_vec = box_xywh_to_xyxy(fack_bbox).view(-1, 4).clamp(min=0.0,max=1.0) 
            real_bbox = box_xywh_to_xyxy(real_bbox).view(-1, 4).clamp(min=0.0,max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)  

            alpha = unit_pix
            eps =alpha*10
            #alpha = (eps * 1.0 / iteration) * scale_factor   #eps最大扰动幅度 iteration迭代几次
            #伪标签

            focal_loss = FocalLoss()
            losses = []
            advs = []
            rgb_target_threshold = 6.6e-5
            rgb_untarget_threshold = 4.5e-6
            
            giou_weight = cfg.TRAIN.GIOU_WEIGHT
            l1_weight = cfg.TRAIN.L1_WEIGHT
            focal_weight = cfg.TRAIN.FOCAL_WEIGHT

            ori_x, ori_aux_dict = self.backbone(
                                        z=template, 
                                        x=ori_rgb_search, 
                                        event_z=event_template, 
                                        event_x=ori_event_search,
                                        ce_template_mask=ce_template_mask,
                                        ce_keep_rate=ce_keep_rate,
                                        return_last_attn=return_last_attn, )
                
            ori_feat_last = ori_x
            if isinstance(ori_x, list):
                ori_feat_last = ori_x[-1]
            ori_out = self.forward_head(ori_feat_last, None) #加显存

            ori_out.update(ori_aux_dict)
            ori_out['backbone_feat'] = ori_x
            ori_pred_dict = ori_out
            ori_pred_boxes = ori_pred_dict['pred_boxes']
            
            if torch.isnan(ori_pred_boxes).any(): # 检查是否有nan
                raise ValueError("Network outputs is NAN! Stop Training")
            ori_pred_boxes_vec = box_cxcywh_to_xyxy(ori_pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)

            with torch.enable_grad():
                for i in range(iteration):
                    self.backbone.zero_grad()
                    self.box_head.zero_grad()
                    previous_adv_search = rgb_adv_search.detach()
                    x, aux_dict = self.backbone(z=template, 
                                                x=rgb_adv_search, 
                                                event_z=event_template, 
                                                event_x=ori_event_search,
                                                ce_template_mask=ce_template_mask,
                                                ce_keep_rate=ce_keep_rate,
                                                return_last_attn=return_last_attn,)
                
                    feat_last = x
                    if isinstance(x, list):
                        feat_last = x[-1]
                    out = self.forward_head(feat_last, None) #加显存

                    out.update(aux_dict)
                    out['backbone_feat'] = x
                    pred_dict = out
                    pred_boxes = pred_dict['pred_boxes']
                    
                    if torch.isnan(pred_boxes).any(): # 检查是否有nan
                        raise ValueError("Network outputs is NAN! Stop Training")
                    pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
    
                    
                    try:
                        rgb_giou_loss_target, iou = giou_loss(pred_boxes_vec, fack_boxes_vec)  # (BN,4) (BN,4)
                        rgb_giou_loss_true, iou = giou_loss(pred_boxes_vec, real_bbox)  # (BN,4) (BN,4)
                        rgb_giou_loss_ori, iou = giou_loss(pred_boxes_vec, ori_pred_boxes_vec)  # (BN,4) (BN,4)
                        rgb_giou_loss_ori_fack, iou = giou_loss(ori_pred_boxes_vec, fack_boxes_vec)  # (BN,4) (BN,4)
                        rgb_giou_loss_ori_ture, iou = giou_loss(ori_pred_boxes_vec, real_bbox)  # (BN,4) (BN,4)
                    except:
                        rgb_giou_loss_target, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
                        rgb_giou_loss_true, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
                        rgb_giou_loss_ori, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
                        rgb_giou_loss_ori_fack, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
                        rgb_giou_loss_ori_ture, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
                    
                    # compute l1 loss
                    rgb_l1_loss_target = l1_loss(pred_boxes_vec, fack_boxes_vec)  # (BN,4) (BN,4) 
                    rgb_l1_loss_true = l1_loss(pred_boxes_vec, real_bbox)  # (BN,4) (BN,4)
                    rgb_l1_loss_ori = l1_loss(pred_boxes_vec, ori_pred_boxes_vec)  # (BN,4) (BN,4) 
                    rgb_l1_loss_ori_fack = l1_loss(ori_pred_boxes_vec, fack_boxes_vec)  # (BN,4) (BN,4)
                    rgb_l1_loss_ori_ture = l1_loss(ori_pred_boxes_vec, real_bbox)  # (BN,4) (BN,4)
                            
                    if 'score_map' in pred_dict:
                        rgb_location_loss_true = focal_loss(pred_dict['score_map'], gt_gaussian_maps)
                        rgb_location_loss_fack = focal_loss(pred_dict['score_map'], fack_gaussian_maps)
                        rgb_location_loss_ori = focal_loss(pred_dict['score_map'], ori_pred_dict['score_map'])
                        rgb_location_loss_ori_fack = focal_loss(ori_pred_dict['score_map'], fack_gaussian_maps)
                        rgb_location_loss_ori_ture = focal_loss(ori_pred_dict['score_map'], gt_gaussian_maps)
                    else:
                        rgb_location_loss_true = torch.tensor(0.0, device=l1_loss.device)
                        rgb_location_loss_fack = torch.tensor(0.0, device=l1_loss.device)
                        rgb_location_loss_ori = torch.tensor(0.0, device=l1_loss.device)
                        rgb_location_loss_ori_fack = torch.tensor(0.0, device=l1_loss.device)
                        rgb_location_loss_ori_ture = torch.tensor(0.0, device=l1_loss.device)
                

                    rgb_loss_giou= -rgb_giou_loss_true -rgb_giou_loss_ori_ture -rgb_giou_loss_ori +rgb_giou_loss_target +rgb_giou_loss_ori_fack
                    rgb_loss_l1= -rgb_l1_loss_true -rgb_l1_loss_ori -rgb_l1_loss_ori_ture +rgb_l1_loss_target +rgb_l1_loss_ori_fack
                    rgb_loss_location = -rgb_location_loss_true -rgb_location_loss_ori -rgb_location_loss_ori_ture + rgb_location_loss_fack +rgb_location_loss_ori_fack
            

                    # loss应当越来越大
                    loss1 =  giou_weight * rgb_loss_giou + l1_weight * rgb_loss_l1 + focal_weight * rgb_loss_location

                    losses.append(loss1.item())
                    self.backbone.zero_grad()
                    self.box_head.zero_grad()
                    if rgb_adv_search.grad is not None:
                        rgb_adv_search.grad.data.fill_(0)
                    
                    # 反向传播 计算参数梯度值 但不会更新参数 更新由梯度下降算法进行
                    # loss1.backward(retain_graph=True)
                    loss1.backward()
                    rgb_adv = rgb_adv_search.grad.detach()
                    adv_grad = where((rgb_adv > rgb_target_threshold ) | (rgb_adv < -rgb_target_threshold), rgb_adv, 0)
                    # adv_grad = torch.sign(adv_grad)
                    adv = alpha * torch.sign(adv_grad)
                    #advs.append(adv.item())
                    rgb_adv_search = rgb_adv_search.detach()
                    adv = adv.detach()
                    with torch.no_grad():
                        rgb_adv_search = rgb_adv_search - adv 
                        # eps 10 限制波动范围
                        #L无穷限制 加显存

                        rgb_adv_search = where(rgb_adv_search > search + eps, search + eps, rgb_adv_search)
                        rgb_adv_search = where(rgb_adv_search < search - eps, search - eps, rgb_adv_search)

                        rgb_adv_search[:,0,:,:] = torch.clamp(rgb_adv_search[:,0,:,:], max=x_val_red_max, min=x_val_red_min)
                        rgb_adv_search[:,1,:,:] = torch.clamp(rgb_adv_search[:,1,:,:], max=x_val_green_max, min=x_val_green_min)
                        rgb_adv_search[:,2,:,:] = torch.clamp(rgb_adv_search[:,2,:,:], max=x_val_blue_max, min=x_val_blue_min)
                        rgb_adv_search = Variable(rgb_adv_search.data, requires_grad=True)

                        # event frame 攻击
                        previous_event_adv_search = event_adv_search
                        # event_adv_search = event_adv_search.clone()
                        event_adv_search = event_adv_search - adv 
                        event_adv_search[:,0,:,:] = torch.clamp(event_adv_search[:,0,:,:], max=event_x_val_red_max, min=event_x_val_red_min)
                        event_adv_search[:,1,:,:] = torch.clamp(event_adv_search[:,1,:,:], max=event_x_val_green_max, min=event_x_val_green_min)
                        event_adv_search[:,2,:,:] = torch.clamp(event_adv_search[:,2,:,:], max=event_x_val_blue_max, min=event_x_val_blue_min)
                    event_adv_search = Variable(event_adv_search.data, requires_grad=True)

                    event_x, event_aux_dict = self.backbone(
                                                z=template, 
                                                x=ori_rgb_search, 
                                                event_z=event_template, 
                                                event_x=event_adv_search,
                                                ce_template_mask=ce_template_mask,
                                                ce_keep_rate=ce_keep_rate,
                                                return_last_attn=return_last_attn, )
                
                    event_feat_last = event_x
                    if isinstance(event_x, list):
                        event_feat_last = event_x[-1]
                    event_out = self.forward_head(event_feat_last, None) #加显存

                    event_out.update(event_aux_dict)
                    event_out['backbone_feat'] = event_x
                    event_pred_dict = event_out
                    event_pred_boxes = event_pred_dict['pred_boxes']
                    
                    if torch.isnan(event_pred_boxes).any(): # 检查是否有nan
                        raise ValueError("Network outputs is NAN! Stop Training")
                    event_pred_boxes_vec = box_cxcywh_to_xyxy(event_pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)

                    try:
                        event_giou_loss_target, iou = giou_loss(event_pred_boxes_vec, fack_boxes_vec)  # (BN,4) (BN,4)
                        event_giou_loss_true, iou = giou_loss(event_pred_boxes_vec, real_bbox)  # (BN,4) (BN,4)
                        event_giou_loss_ori, iou = giou_loss(event_pred_boxes_vec, ori_pred_boxes_vec)  # (BN,4) (BN,4)
                        event_giou_loss_ori_fack, iou = giou_loss(ori_pred_boxes_vec, fack_boxes_vec)  # (BN,4) (BN,4)
                        event_giou_loss_ori_ture, iou = giou_loss(ori_pred_boxes_vec, real_bbox)  # (BN,4) (BN,4)
                    except:
                        event_giou_loss_target, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
                        event_giou_loss_true, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
                        event_giou_loss_ori, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
                        event_giou_loss_ori_fack, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
                        event_giou_loss_ori_ture, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
                    
                    # compute l1 loss
                    event_l1_loss_target = l1_loss(event_pred_boxes_vec, fack_boxes_vec)  # (BN,4) (BN,4) 
                    event_l1_loss_true = l1_loss(event_pred_boxes_vec, real_bbox)  # (BN,4) (BN,4)
                    event_l1_loss_ori = l1_loss(event_pred_boxes_vec, ori_pred_boxes_vec)  # (BN,4) (BN,4) 
                    event_l1_loss_ori_fack = l1_loss(ori_pred_boxes_vec, fack_boxes_vec)  # (BN,4) (BN,4)
                    event_l1_loss_ori_ture = l1_loss(ori_pred_boxes_vec, real_bbox)  # (BN,4) (BN,4)
                            
                    if 'score_map' in event_pred_dict:
                        event_location_loss_true = focal_loss(event_pred_dict['score_map'], gt_gaussian_maps)
                        event_location_loss_fack = focal_loss(event_pred_dict['score_map'], fack_gaussian_maps)
                        event_location_loss_ori = focal_loss(event_pred_dict['score_map'], ori_pred_dict['score_map'])
                        event_location_loss_ori_fack = focal_loss(ori_pred_dict['score_map'], fack_gaussian_maps)
                        event_location_loss_ori_ture = focal_loss(ori_pred_dict['score_map'], gt_gaussian_maps)
                    else:
                        event_location_loss_true = torch.tensor(0.0, device=l1_loss.device)
                        event_location_loss_fack = torch.tensor(0.0, device=l1_loss.device)
                        event_location_loss_ori = torch.tensor(0.0, device=l1_loss.device)
                        event_location_loss_ori_fack = torch.tensor(0.0, device=l1_loss.device)
                        event_location_loss_ori_ture = torch.tensor(0.0, device=l1_loss.device)
                

                    event_loss_giou= -event_giou_loss_true -event_giou_loss_ori_ture -event_giou_loss_ori +event_giou_loss_target +event_giou_loss_ori_fack
                    event_loss_l1= -event_l1_loss_true -event_l1_loss_ori -event_l1_loss_ori_ture +event_l1_loss_target +event_l1_loss_ori_fack
                    event_loss_location = -event_location_loss_true -event_location_loss_ori -event_location_loss_ori_ture + event_location_loss_fack +event_location_loss_ori_fack
            
                    loss2 =  giou_weight * event_loss_giou + l1_weight * event_loss_l1 + focal_weight * event_loss_location

                    losses.append(loss2.item())
                    self.backbone.zero_grad()
                    self.box_head.zero_grad()
                    if event_adv_search.grad is not None:
                        event_adv_search.grad.data.fill_(0)
                    

                    # 反向传播 计算参数梯度值 但不会更新参数 更新由梯度下降算法进行
                    # loss2.backward(retain_graph=True)
                    loss2.backward()
                    current_grad = event_adv_search.grad.detach()
                    adv_grad = where(( current_grad> rgb_target_threshold ) | (current_grad < -rgb_target_threshold), current_grad, 0)
                    # adv_grad = torch.sign(adv_grad)
                    adv = alpha * torch.sign(adv_grad)
                    #advs.append(adv.item())
                    currrent_event_adv_search = event_adv_search.detach()
                    adv = adv.detach()
                    with torch.no_grad():
                        currrent_event_adv_search = currrent_event_adv_search - adv 
                        # 累加
                        currrent_event_adv_search = where(currrent_event_adv_search > event_search + eps, event_search + eps, currrent_event_adv_search)
                        currrent_event_adv_search = where(currrent_event_adv_search < event_search - eps, event_search - eps, currrent_event_adv_search)
                        # 创建新的变量来存储结果，避免原地修改
                        currrent_event_adv_search[:,0,:,:] = torch.clamp(currrent_event_adv_search[:,0,:,:], max=event_x_val_red_max, min=event_x_val_red_min)
                        currrent_event_adv_search[:,1,:,:] = torch.clamp(currrent_event_adv_search[:,1,:,:], max=event_x_val_green_max, min=event_x_val_green_min)
                        currrent_event_adv_search[:,2,:,:] = torch.clamp(currrent_event_adv_search[:,2,:,:], max=event_x_val_blue_max, min=event_x_val_blue_min)            

                    event_adv_search = currrent_event_adv_search.requires_grad_(True)
                    updated_event_adv_search = event_adv_search.detach()
                    # del x, aux_dict, event_x, event_aux_dict
                    # torch.cuda.empty_cache()
                    # if frame_id == 2 or frame_id == 20 or frame_id == 30 or frame_id == 40 or frame_id == 50 or frame_id == 60 or frame_id == 70:
                    #     save_nips_stage2_renew_attack(
                    #                         ori_search=ori_rgb_search,
                    #                         adv = adv,
                    #                         rgb_adv_search = rgb_adv_search,
                    #                         ori_event_search = ori_event_search,
                    #                         event_adv_search = event_adv_search,
                    #                         iteration = 0,
                    #                         frame_id = frame_id,
                    #                         current_time = current_time,
                    #                         base_save_path = '/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c01/DATA/chenqiang/RGB_Event_Frame_Attack/event_CEUTrack/visual/fe108/rgb_guide_frame_attack_senquce',
                    #                         )    

                print("Loss changes over iterations:", losses)    
            return rgb_adv_search,event_adv_search
    

    def forward_head(self, cat_feature, gt_score_map=None):
        """
        cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
       
        # enc_opt1 = cat_feature[:, -self.feat_len_s:] # x 直接删
        # enc_opt2 = cat_feature[:, :self.feat_len_s] # event_x 
        # enc_opt = torch.cat([enc_opt1, enc_opt2], dim=-1)
        #opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        enc_opt1 = cat_feature[:, -self.feat_len_s:] # 
        enc_opt2 = cat_feature[:, :self.feat_len_s] 
        enc_opt = torch.cat([enc_opt1, enc_opt2], dim=-1)
        opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)
        
        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out

        elif self.head_type == "CENTER": #走这里
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,#[1 1 4]
                   'score_map': score_map_ctr,#[1 1 16 16] 中心点得分图 取得分最大的点
                   'size_map': size_map,#[1 2 16 16]  wh
                   'offset_map': offset_map} # [1 2 16 16]  偏移值
            return out
        else:
            raise NotImplementedError


def build_ceutrack(cfg, training=True,noise=None):


    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = "/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c01/DATA/chenqiang/event_attack/event_CEUTrack/pretrained_models"
    if cfg.MODEL.PRETRAIN_FILE and ('CEUTrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''

    if cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224':
        backbone = vit_base_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_ce': #走这
        backbone = vit_base_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                           ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                           ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                           )
        # hidden_dim = backbone.embed_dim
        hidden_dim = backbone.embed_dim * 2 #bbox的初试卷积层
        # hidden_dim = backbone.embed_dim  #bbox的初试卷积层
        patch_start_index = 1

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_large_patch16_224_ce':
        backbone = vit_large_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                            ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                            ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                            )

        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    else:
        raise NotImplementedError

    backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    box_head = build_box_head(cfg, hidden_dim)

    model = CEUTrack(
        backbone,
        box_head,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
        initial_noise=noise
    )

    if 'CEUTrack' in cfg.MODEL.PRETRAIN_FILE and training:
        checkpoint = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)

    return model

def where(cond, x, y):
        """
        code from :
            https://discuss.pytorch.org/t/how-can-i-do-the-operation-the-same-as-np-where/1329/8
        """ #当 cond 为 True（即 1.0）时，选择 x
        cond = cond.float()
        return (cond*x) + ((1-cond)*y)
        
def save_images(rgb_adv_search, adv, updated_adv_search, iteration):
    # 创建保存目录
    save_path = '/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c01/DATA/chenqiang/AttackTracking/CEUTrack/visual/pic/norm'
    folder_path = os.path.join(save_path, f'iteration_{iteration}')
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    rgb_adv_search = (rgb_adv_search - rgb_adv_search.min()) / (rgb_adv_search.max() - rgb_adv_search.min())

    adv = (adv - adv.min()) / (adv.max() - adv.min())

    updated_adv_search = (updated_adv_search - updated_adv_search.min()) / (updated_adv_search.max() - updated_adv_search.min())
    # 保存迭代开始时的 rgb_adv_search
    previous_adv_search = rgb_adv_search[0].detach().cpu().numpy().transpose(1, 2, 0)
    plt.imsave(os.path.join(folder_path, f'Previous_adv_{iteration}.png'), previous_adv_search)

    # 保存当前的 adv
    current_adv = adv[0].detach().cpu().numpy().transpose(1, 2, 0)
    plt.imsave(os.path.join(folder_path, f'adv_{iteration}.png'), current_adv)

    # 保存更新后的 rgb_adv_search
    updated_adv_search = updated_adv_search[0].detach().cpu().numpy().transpose(1, 2, 0)
    plt.imsave(os.path.join(folder_path, f'Current_adv_{iteration}.png'), updated_adv_search)




def save_image_multi_pic(base_save_path,rgb_adv_search, adv, updated_adv_search, iteration, frame_id, current_time, ori_search):
    i = 1
    # 创建保存目录
    # base_save_path = '/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c01/DATA/chenqiang/RGB_Event_Frame_Attack/event_CEUTrack/visual/nips_stage2_renew_attack'
    time_folder_path = os.path.join(base_save_path, current_time)
    if not os.path.exists(time_folder_path):
        os.makedirs(time_folder_path)
    
    # 创建 frame_id 文件夹
    frame_folder_path = os.path.join(time_folder_path, f'frame_{frame_id}')
    if not os.path.exists(frame_folder_path):
        os.makedirs(frame_folder_path)
    
    # 创建 iteration 文件夹
    iteration_folder_path = os.path.join(frame_folder_path, f'iteration_{iteration}')
    if not os.path.exists(iteration_folder_path):
        os.makedirs(iteration_folder_path)
    
    # 归一化处理
    mean = torch.tensor([0.485, 0.456, 0.406]).view((1, 3, 1, 1)).cuda()
    std = torch.tensor([0.229, 0.224, 0.225]).view((1, 3, 1, 1)).cuda()
    rgb_adv_search = rgb_adv_search * std + mean
    adv = adv * std + mean
    updated_adv_search = updated_adv_search * std + mean
    ori_search = ori_search * std + mean
    
    # 将图像从 GPU 移动到 CPU 并转换为 numpy 数组
    rgb_adv_search = rgb_adv_search.squeeze(0).detach().cpu().numpy().transpose((1, 2, 0))
    adv = adv.squeeze(0).detach().cpu().numpy().transpose((1, 2, 0))
    adv = np.clip(adv, 0, 1)
    updated_adv_search = updated_adv_search.squeeze(0).detach().cpu().numpy().transpose((1, 2, 0))
    ori_search = ori_search.squeeze(0).detach().cpu().numpy().transpose((1, 2, 0))  
    # 创建一个包含四个子图的图形
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # 显示每个图像
    axes[0, 0].imshow(rgb_adv_search)
    axes[0, 0].set_title(f'Previous_adv_{iteration}')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(updated_adv_search)
    axes[0, 1].set_title(f'Current_adv_{iteration}')
    axes[0, 1].axis('off')

    axes[1, 0].imshow(adv)
    axes[1, 0].set_title(f'Adv_{iteration}')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(ori_search)
    axes[1, 1].set_title(f'Ori_search_{iteration}')
    axes[1, 1].axis('off')

    # 保存图形
    fig.savefig(os.path.join(iteration_folder_path, f'Combined_{iteration}.png'))
    plt.close(fig)
    
    # 单独保存每个图像
    plt.imsave(os.path.join(iteration_folder_path, f'Previous_adv_{iteration}.png'), rgb_adv_search)
    plt.imsave(os.path.join(iteration_folder_path, f'adv_{iteration}.png'), adv)
    plt.imsave(os.path.join(iteration_folder_path, f'{i}_Ori_search_{iteration}.png'), ori_search)
    plt.imsave(os.path.join(iteration_folder_path, f'Current_adv_{iteration}.png'), updated_adv_search)


def save_nips_stage2_renew_attack(base_save_path,ori_search,adv,rgb_adv_search, ori_event_search,event_adv_search, iteration, frame_id, current_time):
    i = 1
    # 创建保存目录
    time_folder_path = os.path.join(base_save_path, current_time)
    if not os.path.exists(time_folder_path):
        os.makedirs(time_folder_path)
    
    # 创建 frame_id 文件夹
    frame_folder_path = os.path.join(time_folder_path, f'frame_{frame_id}')
    if not os.path.exists(frame_folder_path):
        os.makedirs(frame_folder_path)
    
    # 创建 iteration 文件夹
    iteration_folder_path = os.path.join(frame_folder_path, f'iteration_{iteration}')
    if not os.path.exists(iteration_folder_path):
        os.makedirs(iteration_folder_path)
    
    # 归一化处理
    mean = torch.tensor([0.485, 0.456, 0.406]).view((1, 3, 1, 1)).cuda()
    std = torch.tensor([0.229, 0.224, 0.225]).view((1, 3, 1, 1)).cuda()
    
    ori_search = ori_search * std + mean
    adv = adv * std + mean
    rgb_adv_search = rgb_adv_search * std + mean
    ori_event_search = ori_event_search * std + mean
    event_adv_search = event_adv_search * std + mean

    
    # 将图像从 GPU 移动到 CPU 并转换为 numpy 数组
    ori_search = ori_search.squeeze(0).detach().cpu().numpy().transpose((1, 2, 0))  
    adv = adv.squeeze(0).detach().cpu().numpy().transpose((1, 2, 0))
    adv = np.clip(adv, 0, 1)
    rgb_adv_search = rgb_adv_search.squeeze(0).detach().cpu().numpy().transpose((1, 2, 0))
    ori_event_search = ori_event_search.squeeze(0).detach().cpu().numpy().transpose((1, 2, 0))
    event_adv_search = event_adv_search.squeeze(0).detach().cpu().numpy().transpose((1, 2, 0))
    # 创建一个包含四个子图的图形
    fig, axes = plt.subplots(2, 3, figsize=(12, 12))
    
    # 显示每个图像
    axes[0, 0].imshow(ori_search)
    axes[0, 0].set_title(f'ori_search_{iteration}')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(adv)
    axes[0, 1].set_title(f'adv_{iteration}')
    axes[0, 1].axis('off')
        
    axes[0, 2].imshow(rgb_adv_search)
    axes[0, 2].set_title(f'rgb_adv_search_{iteration}')
    axes[0, 2].axis('off')

    axes[1, 0].imshow(ori_event_search)
    axes[1, 0].set_title(f'ori_event_search_{iteration}')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(adv)
    axes[1, 1].set_title(f'adv_{iteration}')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(event_adv_search)
    axes[1, 2].set_title(f'event_adv_search_{iteration}')
    axes[1, 2].axis('off')

    # 保存图形
    fig.savefig(os.path.join(iteration_folder_path, f'Combined_{iteration}.png'))
    plt.close(fig)
    
    # 单独保存每个图像
    plt.imsave(os.path.join(iteration_folder_path, f'{i}_Ori_search_{iteration}.png'), ori_search)
    plt.imsave(os.path.join(iteration_folder_path, f'adv_{iteration}.png'), adv)
    plt.imsave(os.path.join(iteration_folder_path, f'Rgb_adv_{iteration}.png'), rgb_adv_search)


    plt.imsave(os.path.join(iteration_folder_path, f'{i}_Ori_event_search_{iteration}.png'), ori_event_search)
    plt.imsave(os.path.join(iteration_folder_path, f'Current_adv_{iteration}.png'), event_adv_search)



def save_image_double_pic(base_save_path,rgb_adv_search, adv, updated_adv_search, iteration, frame_id, current_time, ori_search):
    i = 1
    # 创建保存目录
   # base_save_path = '/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c01/DATA/chenqiang/AttackTracking/CEUTrack/visual/double/double_iter'
    # base_save_path = '/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c01/DATA/chenqiang/AttackTracking/CEUTrack/visual/untargeted/iteration/double_attack_1'    

    time_folder_path = os.path.join(base_save_path, current_time)
    if not os.path.exists(time_folder_path):
        os.makedirs(time_folder_path)
    
    # 创建 frame_id 文件夹
    frame_folder_path = os.path.join(time_folder_path, f'frame_{frame_id}')
    if not os.path.exists(frame_folder_path):
        os.makedirs(frame_folder_path)
    
    # 创建 iteration 文件夹
    iteration_folder_path = os.path.join(frame_folder_path, f'iteration_{iteration}')
    if not os.path.exists(iteration_folder_path):
        os.makedirs(iteration_folder_path)
    
    # 归一化处理
    mean = torch.tensor([0.485, 0.456, 0.406]).view((1, 3, 1, 1)).cuda()
    std = torch.tensor([0.229, 0.224, 0.225]).view((1, 3, 1, 1)).cuda()
    rgb_adv_search = rgb_adv_search * std + mean
    adv = adv * std + mean
    updated_adv_search = updated_adv_search * std + mean
    ori_search = ori_search * std + mean
    
    # 将图像从 GPU 移动到 CPU 并转换为 numpy 数组
    rgb_adv_search = rgb_adv_search.squeeze(0).detach().cpu().numpy().transpose((1, 2, 0))
    adv = adv.squeeze(0).detach().cpu().numpy().transpose((1, 2, 0))
    adv = np.clip(adv, 0, 1)
    updated_adv_search = updated_adv_search.squeeze(0).detach().cpu().numpy().transpose((1, 2, 0))
    ori_search = ori_search.squeeze(0).detach().cpu().numpy().transpose((1, 2, 0))  
    # 创建一个包含四个子图的图形
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # 显示每个图像
    axes[0, 0].imshow(rgb_adv_search)
    axes[0, 0].set_title(f'Previous_adv_{iteration}')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(updated_adv_search)
    axes[0, 1].set_title(f'Current_adv_{iteration}')
    axes[0, 1].axis('off')

    axes[1, 0].imshow(adv)
    axes[1, 0].set_title(f'Adv_{iteration}')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(ori_search)
    axes[1, 1].set_title(f'Ori_search_{iteration}')
    axes[1, 1].axis('off')

    # 保存图形
    fig.savefig(os.path.join(iteration_folder_path, f'Combined_{iteration}.png'))
    plt.close(fig)
    
    # 单独保存每个图像
    plt.imsave(os.path.join(iteration_folder_path, f'Previous_adv_{iteration}.png'), rgb_adv_search)
    plt.imsave(os.path.join(iteration_folder_path, f'adv_{iteration}.png'), adv)
    plt.imsave(os.path.join(iteration_folder_path, f'{i}_Ori_search_{iteration}.png'), ori_search)
    plt.imsave(os.path.join(iteration_folder_path, f'Current_adv_{iteration}.png'), updated_adv_search)


