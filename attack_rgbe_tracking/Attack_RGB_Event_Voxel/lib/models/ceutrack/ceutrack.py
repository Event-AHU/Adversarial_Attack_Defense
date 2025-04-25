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
from torch.nn.functional import gumbel_softmax
from lib.gumbel_softmax.tracking_gumbel.HardDiffArgmax1 import HardDiffArgmax

# logging.basicConfig(
#     filename='/wangx/DATA/Code/chenqiang/Attack/RGB_Event_Voxel_Attack/AttackTracking/CEUTrack/output/test/tracking_results/log',
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     datefmt='%Y-%m-%d %H:%M:%S'
# )

class CEUTrack(nn.Module):
    """ This is the base class for ceutrack """

    def __init__(self, transformer, box_head, aux_loss=False, head_type="CORNER"):
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
        
        self.gumbel_alpha = nn.Parameter(torch.randn(1, 4096, 3))
        self.alpha_optimizer = torch.optim.Adam([self.gumbel_alpha], lr=1,betas=(0.9, 0.999),eps=1e-08,weight_decay=0)
        self.HardDiffArgmax = HardDiffArgmax()

        # self.gmbel_alpha = torch.nn.Parameter(torch.randn(initial_shape, requires_grad=True))
        # self.alpha_optimizer = torch.optim.Adam([self.gmbel_alpha], lr=1e-3)


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
    
    def forward_attck_img(self, template: torch.Tensor,
                search: torch.Tensor,
                ori_search: torch.Tensor,
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
        alpha = unit_pix
        eps =alpha*10
        #alpha = (eps * 1.0 / iteration) * scale_factor   #eps最大扰动幅度 iteration迭代几次
        #伪标签
        fack_bbox = adv_bbox 
        focal_loss = FocalLoss()
        losses = []
        advs = []
        rgb_target_threshold = 6.6e-5
        rgb_untarget_threshold = 4.5e-6
         # 记录函数参数
        # logging.info(f"""Function parameters: 
        # eps={eps}, 
        # alpha={alpha}, 
        # iteration={iteration}, 
        # current_time={current_time}""")
        with torch.enable_grad():
            ori_x, ori_aux_dict = self.backbone(z=template, x=ori_search, event_z=event_template, 
                                            event_x=event_search,
                                            ce_template_mask=ce_template_mask,
                                            ce_keep_rate=ce_keep_rate,
                                            return_last_attn=return_last_attn, )
            # 前向头部
            ori_feat_last = ori_x
            if isinstance(ori_x, list):
                ori_feat_last = ori_x[-1]
            ori_out = self.forward_head(ori_feat_last, None)

            ori_out.update(ori_aux_dict)
            ori_out['backbone_feat'] = ori_x
            ori_pred_dict = ori_out
            ori_pred_boxes = ori_pred_dict['pred_boxes']
            if torch.isnan(ori_pred_boxes).any():  # 检查是否有nan
                raise ValueError("Network outputs is NAN! Stop Training")
            ori_pred_boxes_vec = box_cxcywh_to_xyxy(ori_pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
                            

            for i in range(iteration):
                previous_adv_search = rgb_adv_search.clone()
                x, aux_dict = self.backbone(z=template, x=rgb_adv_search, 
                                            event_z=event_template, 
                                            event_x=event_search,
                                            ce_template_mask=ce_template_mask,
                                            ce_keep_rate=ce_keep_rate,
                                            return_last_attn=return_last_attn, )
            
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
                fack_boxes_vec = box_xywh_to_xyxy(fack_bbox).view(-1, 4).clamp(min=0.0,max=1.0) 
                real_bbox = box_xywh_to_xyxy(real_bbox).view(-1, 4).clamp(min=0.0,max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)    
                 
                try:
                    giou_loss_target, iou = giou_loss(pred_boxes_vec, fack_boxes_vec)  # (BN,4) (BN,4)
                    giou_loss_true, iou = giou_loss(pred_boxes_vec, real_bbox)  # (BN,4) (BN,4)
                    giou_loss_ori, iou = giou_loss(pred_boxes_vec, ori_pred_boxes_vec)  # (BN,4) (BN,4)
                    giou_loss_ori_fack, iou = giou_loss(ori_pred_boxes_vec, fack_boxes_vec)  # (BN,4) (BN,4)
                    giou_loss_ori_ture, iou = giou_loss(ori_pred_boxes_vec, real_bbox)  # (BN,4) (BN,4)

                except:
                    giou_loss_target, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
                    giou_loss_true, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
                    giou_loss_ori, iou =  torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
                    giou_loss_ori_fack, iou =  torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
                    giou_loss_ori_ture, iou =  torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()

                # 计算 l1 loss 
                if 'score_map' in pred_dict:
                    location_loss_true = focal_loss(pred_dict['score_map'], gt_gaussian_maps)
                    location_loss_fack = focal_loss(pred_dict['score_map'], fack_gaussian_maps)
                    location_loss_ori = focal_loss(pred_dict['score_map'], ori_pred_dict['score_map'])
                    location_loss_ori_fack = focal_loss(ori_pred_dict['score_map'], fack_gaussian_maps)
                    location_loss_ori_ture = focal_loss(ori_pred_dict['score_map'], gt_gaussian_maps)
                else:
                    location_loss_true = torch.tensor(0.0, device=l1_loss.device)
                    location_loss_fack = torch.tensor(0.0, device=l1_loss.device)
                    location_loss_ori = torch.tensor(0.0, device=l1_loss.device)
                    location_loss_ori_fack = torch.tensor(0.0, device=l1_loss.device)
                    location_loss_ori_ture = torch.tensor(0.0, device=l1_loss.device)


                l1_loss_target = l1_loss(pred_boxes_vec, fack_boxes_vec)  # (BN,4) (BN,4)
                l1_loss_true = l1_loss(pred_boxes_vec, real_bbox)  # (BN,4) (BN,4)
                l1_loss_ori = l1_loss(pred_boxes_vec, ori_pred_boxes_vec)  # (BN,4) (BN,4)
                l1_loss_ori_fack = l1_loss(ori_pred_boxes_vec, fack_boxes_vec)  # (BN,4) (BN,4)
                l1_loss_ori_ture = l1_loss(ori_pred_boxes_vec, real_bbox)  # (BN,4) (BN,4)

                loss_giou = -giou_loss_true -giou_loss_ori -giou_loss_ori_ture + giou_loss_target +giou_loss_ori_fack
                loss_l1 = -l1_loss_true -l1_loss_ori -l1_loss_ori_ture + l1_loss_target +l1_loss_ori_fack
                loss_location = -location_loss_true -location_loss_ori -location_loss_ori_ture + location_loss_fack +location_loss_ori_fack
                
                # loss_giou = -giou_loss_true -giou_loss_ori + giou_loss_target 
                # loss_l1 = -l1_loss_true -l1_loss_ori + l1_loss_target 
                # loss_location = -location_loss_true -location_loss_ori + location_loss_fack 
                                
                
                # # loss_giou=  -giou_loss_true
                # loss_l1=  -l1_loss_true
                # loss_location =  -location_loss_true
                # weighted sum 1 14 1
                giou_weight = cfg.TRAIN.GIOU_WEIGHT
                l1_weight = cfg.TRAIN.L1_WEIGHT
                focal_weight = cfg.TRAIN.FOCAL_WEIGHT
                # loss应当越来越大
                loss =  giou_weight * loss_giou + l1_weight * loss_l1 + focal_weight * loss_location

                # 记录每次迭代的损失值
                # logging.info(f"""Iteration {i}: 
                #                 loss_giou={loss_giou.item()}, 
                #                 loss_l1={loss_l1.item()}, 
                #                 loss_location={loss_location.item()}, 
                #                 total_loss={loss.item()}""")
               
                #print("loss:{:.4f},loss_giou: {:.4f}, loss_l1: {:.4f}, loss_location: {:.4f}".format(loss.item(),loss_giou.item(), loss_l1.item(), loss_location.item()))
                losses.append(loss.item())
                self.backbone.zero_grad()
                self.box_head.zero_grad()
                if rgb_adv_search.grad is not None:
                    rgb_adv_search.grad.data.fill_(0)
                
                rgb_adv_search.retain_grad()  # 保留 x_adv 的梯度
                # 反向传播 计算参数梯度值 但不会更新参数 更新由梯度下降算法进行
                loss.backward(retain_graph=True)
                # if rgb_adv_search.grad is None:
                #     print("x_adv.grad is None")
                #     continue
                # 梯度存储在grad中 
                # 过滤无意义的梯度
                # 梯度越大 损失越大 梯度为负数 越应该减小其值
                # 保留过大或者过小的梯度 按照标准差缩放
                
                adv_grad = where((rgb_adv_search.grad > rgb_target_threshold ) | (rgb_adv_search.grad < -rgb_target_threshold), rgb_adv_search.grad, 0)
                adv_grad = torch.sign(adv_grad)
                adv = alpha * adv_grad
                #advs.append(adv.item())
                rgb_adv_search = rgb_adv_search - adv 
                # eps 10 限制波动范围
                #L无穷限制 加显存
                rgb_adv_search = rgb_adv_search.detach().clone()
                rgb_adv_search = where(rgb_adv_search > search + eps, search + eps, rgb_adv_search)
                rgb_adv_search = where(rgb_adv_search < search - eps, search - eps, rgb_adv_search)
                
                # 创建新的变量来存储结果，避免原地修改
                new_rgb_adv_search = rgb_adv_search.clone()
                new_rgb_adv_search[:,0,:,:] = torch.clamp(new_rgb_adv_search[:,0,:,:], max=x_val_red_max, min=x_val_red_min)
                new_rgb_adv_search[:,1,:,:] = torch.clamp(new_rgb_adv_search[:,1,:,:], max=x_val_green_max, min=x_val_green_min)
                new_rgb_adv_search[:,2,:,:] = torch.clamp(new_rgb_adv_search[:,2,:,:], max=x_val_blue_max, min=x_val_blue_min)
                rgb_adv_search = new_rgb_adv_search.detach().clone().requires_grad_(True)

                updated_adv_search = rgb_adv_search
 
                # #if 1 <= frame_id <= 600:
                # if frame_id == 3 or frame_id == 300 or frame_id == 500:
                #     save_image_multi_pic(rgb_adv_search = previous_adv_search,
                #                          adv = adv,
                #                          updated_adv_search = updated_adv_search,
                #                          iteration = i,
                #                          frame_id = frame_id,
                #                          current_time = current_time,
                #                          ori_search=ori_rgb_search)
              #  if frame_id == 61:
                    #print("frame_id reached 61, terminating the program.")
                   # sys.exit(0)  # 中止程序
            print("Rgb Loss changes over iterations:", losses)    
            # del x, aux_dict, adv, updated_adv_search, ori_rgb_search
            # torch.cuda.empty_cache()   
        return rgb_adv_search
    
    
    def forward_null_event_shift_xyt(self, template: torch.Tensor,
                            search: torch.Tensor,
                            ori_search: torch.Tensor,
                            event_template: torch.Tensor,        # torch.Size([4, 1, 19, 10000])
                            event_search: torch.Tensor,          # torch.Size([4, 1, 19, 10000])
                            ce_template_mask=None,
                            ce_keep_rate=None,
                            return_last_attn=False,
                            eps=0.2, 
                            alpha=0.1, 
                            iteration=10, 
                            time_val_min=0, 
                            time_val_max=1,
                            unit_time = 0,
                            unit_x = 0,
                            unit_y = 0,
                            adv_bbox = None,
                            att_t = 0,
                            att_x = 0,
                            att_y = 0,
                            cfg = None,
                            frame_id = 0,
                            current_time = None,
                            real_bbox= None,
                            gt_gaussian_maps = None,
                            fack_gaussian_maps = None,
                            # null_event = None
                            ):

            print("iteration: ",iteration)               
            origin_event_search = event_search
            coordinate_min =0
            coordinate_max = 1
            x_ori = origin_event_search[:,:,0,:]
            y_ori = origin_event_search[:,:,1,:]
            time_ori = origin_event_search[:,:,2,:]
        
            event_adv_search = event_search.clone()
            time_adv = time_ori.clone()
            x_adv = x_ori.clone()
            y_adv = y_ori.clone()

            scale_factor = 0.01
            scale_factor_new = 1
            # 继承上一帧的扰动
            time_adv = time_adv + att_t  # 初试为0
            x_adv = x_adv + att_x   # 初试为0
            y_adv = y_adv + att_y   # 初试为0

            # 上一帧扰动不可太大
            x_adv = torch.clamp(x_adv, min=coordinate_min, max=coordinate_max)
            y_adv = torch.clamp(y_adv, min=coordinate_min, max=coordinate_max)
            time_adv = torch.clamp(time_adv, min=time_val_min, max=time_val_max)
            
            #继承上一帧扰动
            event_adv_search[:,:,0,:] = x_adv
            event_adv_search[:,:,1,:] = y_adv
            event_adv_search[:,:,2,:] = time_adv
            event_adv_search = Variable(event_adv_search.data, requires_grad=True) 
            
            #限制范围
            eps_time = 8 * unit_time
            alpha_time = 4* unit_time  
            
            eps_x = 8 * unit_x
            alpha_x = 4 * unit_x
            
            eps_y = 8 * unit_y
            alpha_y = 4 * unit_y
            
            fack_bbox = adv_bbox         
            focal_loss = FocalLoss()

            losses = []
            
            with torch.enable_grad():   
                ori_x, ori_aux_dict = self.backbone(z=template, x=ori_search, event_z=event_template, 
                                                event_x=origin_event_search,
                                                ce_template_mask=ce_template_mask,
                                                ce_keep_rate=ce_keep_rate,
                                                return_last_attn=return_last_attn, )
                # 前向头部
                ori_feat_last = ori_x
                if isinstance(ori_x, list):
                    ori_feat_last = ori_x[-1]
                ori_out = self.forward_head(ori_feat_last, None)

                ori_out.update(ori_aux_dict)
                ori_out['backbone_feat'] = ori_x
                ori_pred_dict = ori_out
                ori_pred_boxes = ori_pred_dict['pred_boxes']
                if torch.isnan(ori_pred_boxes).any():  # 检查是否有nan
                    raise ValueError("Network outputs is NAN! Stop Training")
                ori_pred_boxes_vec = box_cxcywh_to_xyxy(ori_pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)


                # rgb_adv guide
                rgb_adv_x, rgb_adv_aux_dict = self.backbone(
                                                z=template, x=search,
                                                event_z=event_template, 
                                                event_x=origin_event_search,
                                                ce_template_mask=ce_template_mask,
                                                ce_keep_rate=ce_keep_rate,
                                                return_last_attn=return_last_attn, )
                # 前向头部
                rgb_adv_feat_last = rgb_adv_x
                if isinstance(rgb_adv_x, list):
                    rgb_adv_feat_last = rgb_adv_x[-1]
                rgb_adv_out = self.forward_head(rgb_adv_feat_last, None)

                rgb_adv_out.update(rgb_adv_aux_dict)
                rgb_adv_out['backbone_feat'] = rgb_adv_x
                rgb_adv_pred_dict = rgb_adv_out
                rgb_adv_pred_boxes = rgb_adv_pred_dict['pred_boxes']
                if torch.isnan(rgb_adv_pred_boxes).any():  # 检查是否有nan
                    raise ValueError("Network outputs is NAN! Stop Training")
                rgb_adv_pred_boxes_vec = box_cxcywh_to_xyxy(rgb_adv_pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
                                    



                for i in range(iteration):
                    # 保存最原始的event数据
                    previous_adv_search = event_adv_search.clone()

                    x, aux_dict = self.backbone(z=template, x=search, event_z=event_template, 
                                                event_x=event_adv_search,
                                                ce_template_mask=ce_template_mask,
                                                ce_keep_rate=ce_keep_rate,
                                                return_last_attn=return_last_attn, )
                
                    # 前向头部
                    feat_last = x
                    if isinstance(x, list):
                        feat_last = x[-1]
                    out = self.forward_head(feat_last, None)

                    out.update(aux_dict)
                    out['backbone_feat'] = x
                    pred_dict = out
                    pred_boxes = pred_dict['pred_boxes']
                    
                    if torch.isnan(pred_boxes).any():  # 检查是否有nan
                        raise ValueError("Network outputs is NAN! Stop Training")
                    pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
                    fack_boxes_vec = box_xywh_to_xyxy(fack_bbox).view(-1, 4).clamp(min=0.0, max=1.0) 
                    real_bbox = box_xywh_to_xyxy(real_bbox).view(-1, 4).clamp(min=0.0, max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)    
                    # 计算 giou 和 iou
                    try:
                        giou_loss_target, iou = giou_loss(pred_boxes_vec, fack_boxes_vec)  # (BN,4) (BN,4)
                        giou_loss_true, iou = giou_loss(pred_boxes_vec, real_bbox)  # (BN,4) (BN,4)
                        giou_loss_ori, iou = giou_loss(pred_boxes_vec, ori_pred_boxes_vec)  # (BN,4) (BN,4)
                        giou_loss_ori_fack, iou = giou_loss(ori_pred_boxes_vec, fack_boxes_vec)  # (BN,4) (BN,4)
                        giou_loss_ori_ture, iou = giou_loss(ori_pred_boxes_vec, real_bbox)  # (BN,4) (BN,4)
                        giou_loss_rgb_adv, iou = giou_loss(rgb_adv_pred_boxes_vec, pred_boxes_vec)  # (BN,4) (BN,4)

                    except:
                        giou_loss_target, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
                        giou_loss_true, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
                        giou_loss_ori, iou =  torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
                        giou_loss_ori_fack, iou =  torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
                        giou_loss_ori_ture, iou =  torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
                        giou_loss_rgb_adv, iou =  torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()

                    # 计算 l1 loss 
                    if 'score_map' in pred_dict:
                        location_loss_true = focal_loss(pred_dict['score_map'], gt_gaussian_maps)
                        location_loss_fack = focal_loss(pred_dict['score_map'], fack_gaussian_maps)
                        location_loss_ori = focal_loss(pred_dict['score_map'], ori_pred_dict['score_map'])
                        location_loss_ori_fack = focal_loss(pred_dict['score_map'], ori_pred_dict['score_map'])
                        location_loss_ori_ture = focal_loss(pred_dict['score_map'], ori_pred_dict['score_map'])
                    else:
                        location_loss_true = torch.tensor(0.0, device=l1_loss.device)
                        location_loss_fack = torch.tensor(0.0, device=l1_loss.device)
                        location_loss_ori = torch.tensor(0.0, device=l1_loss.device)
                        location_loss_ori_fack = torch.tensor(0.0, device=l1_loss.device)
                        location_loss_ori_ture = torch.tensor(0.0, device=l1_loss.device)


                    l1_loss_target = l1_loss(pred_boxes_vec, fack_boxes_vec)  # (BN,4) (BN,4)
                    l1_loss_true = l1_loss(pred_boxes_vec, real_bbox)  # (BN,4) (BN,4)
                    l1_loss_ori = l1_loss(pred_boxes_vec, ori_pred_boxes_vec)  # (BN,4) (BN,4)
                    l1_loss_ori_fack = l1_loss(ori_pred_boxes_vec, fack_boxes_vec)  # (BN,4) (BN,4)
                    l1_loss_ori_ture = l1_loss(ori_pred_boxes_vec, real_bbox)  # (BN,4) (BN,4)

                    loss_giou = -giou_loss_true -giou_loss_ori -giou_loss_ori_ture + giou_loss_target+giou_loss_ori_fack
                    loss_l1 = -l1_loss_true -l1_loss_ori -l1_loss_ori_ture + l1_loss_target +l1_loss_ori_fack
                    loss_location = -location_loss_true -location_loss_ori -location_loss_ori_ture + location_loss_fack +location_loss_ori_fack

                    # loss_giou=  -giou_loss_true
                    # loss_l1=  -l1_loss_true
                    # loss_location =  -location_loss_true    
                    # 加权求和
                    giou_weight = cfg.TRAIN.GIOU_WEIGHT
                    l1_weight = cfg.TRAIN.L1_WEIGHT
                    focal_weight = cfg.TRAIN.FOCAL_WEIGHT
                    loss =  giou_weight * loss_giou + l1_weight * loss_l1 + focal_weight * loss_location
                    # 前向传播
                    losses.append(loss.item())
                    # 反向传播 计算参数梯度值 但不会更新参数 更新由梯度下降算法进行
                    self.backbone.zero_grad()
                    self.box_head.zero_grad()
                    if event_adv_search.grad is not None:
                        event_adv_search.grad.data.fill_(0)
                    
                    loss.backward(retain_graph=True)  
                    event_adv_search_grad = event_adv_search.grad.detach().clone()  
                    
                    if event_adv_search_grad is None:
                        print("event_adv_search.grad is None")
                    # 获取 time_adv 的梯度
                    time_adv_grad = event_adv_search_grad[:,:,2,:].clone()
                    x_adv_grad = event_adv_search_grad[:,:,0,:].clone()
                    y_adv_grad = event_adv_search_grad[:,:,1,:].clone()

                    # 保留重要的梯度点 标准差4倍
                    t_adv_grad = torch.where((time_adv_grad > 3e-6 ) | (time_adv_grad < -3e-6), time_adv_grad, 0.0)
                    t_adv_grad = torch.sign(t_adv_grad)
                    t_adv = alpha_time * t_adv_grad
                    new_time_adv = time_adv - t_adv
                    
                    x_adv_grad = torch.where((x_adv_grad > 3e-6 ) | (x_adv_grad < -3e-6), x_adv_grad, 0.0)
                    x_adv_grad = torch.sign(x_adv_grad)
                    x_adv_sign = alpha_x * x_adv_grad
                    new_x_adv = x_adv - x_adv_sign

                    y_adv_grad = torch.where((y_adv_grad > 3e-6 ) | (y_adv_grad < -3e-6), y_adv_grad, 0.0)
                    y_adv_grad = torch.sign(y_adv_grad)
                    y_adv_sign = alpha_y * y_adv_grad
                    new_y_adv = x_adv - y_adv_sign


                    # L无穷限制 
                    new_time_adv = new_time_adv.detach().clone()
                    new_time_adv = torch.where(new_time_adv > time_ori + eps_time, time_ori + eps_time, new_time_adv)
                    new_time_adv = torch.where(new_time_adv < time_ori - eps_time, time_ori - eps_time, new_time_adv)
                    new_time_adv = torch.clamp(new_time_adv, min=time_val_min, max=time_val_max)
                    
                    new_x_adv = new_x_adv.detach().clone()
                    new_x_adv = torch.where(new_x_adv > x_ori + eps_x, x_ori + eps_x, new_x_adv)
                    new_x_adv = torch.where(new_x_adv < x_ori - eps_x, x_ori - eps_x, new_x_adv)
                    new_x_adv = torch.clamp(new_x_adv, min=coordinate_min, max=coordinate_max)
    
                    new_y_adv = new_y_adv.detach().clone()
                    new_y_adv = torch.where(new_y_adv > y_ori + eps_y, y_ori + eps_y, new_time_adv)
                    new_y_adv = torch.where(new_y_adv < y_ori - eps_y, y_ori - eps_y, new_time_adv)
                    new_y_adv = torch.clamp(new_y_adv, min=coordinate_min, max=coordinate_max)

                    # 创建一个新的 event_adv_search 变量来避免原地操作
                    new_event_adv_search = event_adv_search.clone()
                    new_event_adv_search[:,:,2,:] = new_time_adv
                    new_event_adv_search[:,:,0,:] = new_x_adv
                    new_event_adv_search[:,:,1,:] = new_y_adv                
                    
                    event_adv_search = new_event_adv_search.detach().requires_grad_(True)
                    updated_adv_search = event_adv_search
                    # 保存图片
                    # save_images(previous_adv_search, adv, updated_adv_search, i)  
                    # if 1 <= frame_id <= 60:
                    #    save_image_multi_pic(rgb_adv_search = previous_adv_search,adv = adv,updated_adv_search = updated_adv_search,iteration = i,frame_id = frame_id,current_time = current_time,ori_search=ori_search)
                    #  if frame_id == 61:
                            # print("frame_id reached 61, terminating the program.")
                        # sys.exit(0)  # 中止程序
            
            
            print("Loss changes over iterations:", losses)    

            return event_adv_search, new_time_adv, new_x_adv, new_y_adv


    def forward_head(self, cat_feature, gt_score_map=None):
        """
        cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        # output the last 256
        # enc_opt = cat_feature[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)  768*256
        # opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        # bs, Nq, C, HW = opt.size()
        # opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

        ## dual head   768+768)*256
        enc_opt1 = cat_feature[:, -self.feat_len_s:]
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

        elif self.head_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,#[1 1 4]
                   'score_map': score_map_ctr,#[1 1 16 16] 16*8 对应1个8*8的置信度
                   'size_map': size_map,#[1 2 16 16] 2表示宽高 1*8 对应1个8*8的宽高
                   'offset_map': offset_map} # [1 2 16 16] 偏移点
            return out
        else:
            raise NotImplementedError


def build_ceutrack(cfg, training=True):

    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    # pretrained_path = os.path.join(current_dir, '../../../pretrained_models')
    pretrained_path = "/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c01/DATA/chenqiang/AttackTracking/CEUTrack/pretrained_models"
    if cfg.MODEL.PRETRAIN_FILE and ('CEUTrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''

    if cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224':
        backbone = vit_base_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_ce':
        backbone = vit_base_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                           ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                           ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                           )
        # hidden_dim = backbone.embed_dim
        hidden_dim = backbone.embed_dim * 2
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
    save_path = '/wangx/DATA/Code/chenqiang/Attack/RGB_Event_Voxel_Attack/AttackTracking/CEUTrack/visual/pic/norm'
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




def save_image_multi_pic(rgb_adv_search, adv, updated_adv_search, iteration, frame_id, current_time, ori_search):
    i = 1
    # 创建保存目录
    base_save_path = '/wangx/DATA/Code/chenqiang/Attack/RGB_Event_Voxel_Attack/AttackTracking/CEUTrack/visual/untargeted/iteration/rgb_attack_1'
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


def save_image_double_pic(rgb_adv_search, adv, updated_adv_search, iteration, frame_id, current_time, ori_search):
    i = 1
    # 创建保存目录
   # base_save_path = '/wangx/DATA/Code/chenqiang/Attack/RGB_Event_Voxel_Attack/AttackTracking/CEUTrack/visual/double/double_iter'
    base_save_path = '/wangx/DATA/Code/chenqiang/Attack/RGB_Event_Voxel_Attack/AttackTracking/CEUTrack/visual/untargeted/iteration/double_attack_1'    

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


