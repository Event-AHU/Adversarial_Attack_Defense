import math

from lib.models.ceutrack import build_ceutrack
from lib.test.tracker.basetracker import BaseTracker
import torch
import copy
import datetime
from lib.test.tracker.vis_utils import gen_visualization
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target
# for debug
import cv2
import os
import torch.nn.functional as F
from lib.test.tracker.data_utils import Preprocessor
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond
import numpy as np
from lib.train.data.processing_utils import transform_image_to_crop
from lib.utils.heapmap_utils import generate_heatmap
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torch.distributions as distributions
import random

class CEUTrack(BaseTracker):
    def __init__(self, params, dataset_name,noise=None):
        super(CEUTrack, self).__init__(params)
        network = build_ceutrack(params.cfg, training=False,noise=noise)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=False)
        print('load model from {}'.format(self.params.checkpoint))
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor() #处理数据
        self.state = None
        self.dataset_name = dataset_name
        self.noise = noise
        # 16
        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        # motion constrain [1 1 16 16]
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()
        self.current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # for debug
        self.debug = params.debug
        self.use_visdom = params.debug
        self.frame_id = 0

        if self.debug:
            if not self.use_visdom:
                self.save_dir = "debug"
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
            else:
                # self.add_hook()
                self._init_visdom(None, 1)
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes
        self.z_dict1 = {}

    def initialize(self, image, event_image,  info: dict, idx=0):
        # forward the template once # 模版图像 128 *128 缩放大小 有效区域 模版原始坐标

        # 1. sample the template patch
        event_z_patch_arr, resize_factor, event_z_amask_arr, crop_coor = sample_target(event_image, info['init_bbox'],
                                                                           self.params.template_factor, #2
                                                                           output_sz=self.params.template_size) # 128
        
        event_template = self.preprocessor.process(event_z_patch_arr, event_z_amask_arr) # template是对象 包含归一化后的图像和mask
        self.event_template = event_template

        z_patch_arr, resize_factor, z_amask_arr, crop_coor = sample_target(image, info['init_bbox'],
                                                                           self.params.template_factor, #2
                                                                           output_sz=self.params.template_size) # 128
        self.z_patch_arr = z_patch_arr
        template = self.preprocessor.process(z_patch_arr, z_amask_arr) # template是对象 包含归一化后的图像和mask
        
        
        with torch.no_grad():
            self.z_dict1 = template

        self.box_mask_z = None
        if self.cfg.MODEL.BACKBONE.CE_LOC: #[3 6 9]
            template_bbox = self.transform_bbox_to_crop(info['init_bbox'], resize_factor,
                                                        template.tensors.device).squeeze(1) #将初始边界框从原始图像坐标转换到裁剪图像坐标，并将其移动到与模板张量相同的设备上
            self.box_mask_z = generate_mask_cond(self.cfg, 1, template.tensors.device, template_bbox)
        
        # save states
        self.state = info['init_bbox'] #边界框
        self.frame_id = idx #0
        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}
    #跟一帧
    def track(self, image, event_img,att_per,att_per_rgb,
                    att_per_event,real_bbox, 
                    att_x,att_y,att_t,noise=None,attack="rgb",
                    senqice_name=None,
                    info: dict = None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        H, W, _ = image.shape #260 346
        self.frame_id += 1

        event_patch_arr, resize_factor, event_amask_arr, crop_coor = sample_target(event_img, self.state, self.params.search_factor, # 4
                                                                output_sz=self.params.search_size)   # (x1, y1, w, h)
        
        x_patch_arr, resize_factor, x_amask_arr, crop_coor = sample_target(image, self.state, self.params.search_factor, # 4
                                                                output_sz=self.params.search_size)   # (x1, y1, w, h)
        output_sz=self.params.search_size
        crop_sz = torch.Tensor([output_sz, output_sz]).to(device)
       
        search = self.preprocessor.process(x_patch_arr, x_amask_arr).to(device)
        event_search = self.preprocessor.process(event_patch_arr, event_amask_arr).to(device)
        self.event_template.to(device)
        
        real_bbox = real_bbox.reshape(-1)
        bbox_ect = torch.tensor(self.state, dtype=torch.float).to(device)
        real_bbox_norm = transform_image_to_crop(real_bbox,bbox_ect, resize_factor,crop_sz,normalize=True)
        bs = self.cfg.TRAIN.BATCH_SIZE
        real_bbox_map = real_bbox_norm.unsqueeze(0).unsqueeze(0).repeat(bs, 1, 1)
        real_bbox_map = real_bbox_map.permute(1,0,2)
        #real_bbox_map = real_bbox_norm[-1].repeat(bs, 1)  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
        gt_gaussian_maps = generate_heatmap(real_bbox_map, self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.BACKBONE.STRIDE)
        gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1) #[bs 1 16 16]
        
        rate_xy1 = self.cfg.ATX.X
        rate_xy2 = self.cfg.ATX.Y
        rate_wd1 = self.cfg.ATX.W
        rate_wd2 = self.cfg.ATX.H
        fack_bbox = [rate_xy1, rate_xy2, rate_wd1, rate_wd2] # [x y w h]
        fack_bbox_tensor = torch.tensor(fack_bbox)
        fack_bbox_tensor = fack_bbox_tensor.to(device) #[bs 1 4]
        fack_bbox_norm = transform_image_to_crop(fack_bbox_tensor,bbox_ect, resize_factor,crop_sz,normalize=True)
        fack_bbox_map = fack_bbox_norm.unsqueeze(0).unsqueeze(0).repeat(bs, 1, 1)
        fack_bbox_map = fack_bbox_map.permute(1,0,2)
        fack_gaussian_maps = generate_heatmap(fack_bbox_map, self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.BACKBONE.STRIDE)
        fack_gaussian_maps = fack_gaussian_maps[-1].unsqueeze(1)
        
        fack_bbox = fack_bbox_norm
        attack = self.cfg.ATX.attack


        with torch.no_grad():
            x_dict = search
            ones_tensor = torch.ones_like(x_dict.tensors)
            zero_tensor = torch.zeros_like(x_dict.tensors)
            mean = torch.tensor([0.485, 0.456, 0.406]).view((1, 3, 1, 1)).cuda()
            std = torch.tensor([0.229, 0.224, 0.225]).view((1, 3, 1, 1)).cuda()
            ones_tensor = ((ones_tensor / 255.0) - mean) / std 
            zero_tensor = ((zero_tensor / 255.0) - mean) / std 
            unit_pix = (ones_tensor-zero_tensor).to(device)
            self.network.unix = unit_pix
            self.max_eps = 10 * unit_pix
            self.network.max_eps =  10 * self.network.unix
            self.network.optimizer.param_groups[0]["lr"] = self.max_eps / 10
            # 针对rgb的限制 缺个event的限制
            x_val_red_max = torch.max(x_dict.tensors[:,0,:,:]).to(device)
            x_val_red_min = torch.min(x_dict.tensors[:,0,:,:]).to(device)
            x_val_green_max = torch.max(x_dict.tensors[:,1,:,:]).to(device)
            x_val_green_min = torch.min(x_dict.tensors[:,1,:,:]).to(device)
            x_val_blue_max = torch.max(x_dict.tensors[:,2,:,:]).to(device)
            x_val_blue_min = torch.min(x_dict.tensors[:,2,:,:]).to(device)
            
            event_x_val_red_max = torch.max(event_search.tensors[:,0,:,:]).to(device)
            event_x_val_red_min = torch.min(event_search.tensors[:,0,:,:]).to(device)
            event_x_val_green_max = torch.max(event_search.tensors[:,1,:,:]).to(device)
            event_x_val_green_min = torch.min(event_search.tensors[:,1,:,:]).to(device)
            event_x_val_blue_max = torch.max(event_search.tensors[:,2,:,:]).to(device)
            event_x_val_blue_min = torch.min(event_search.tensors[:,2,:,:]).to(device)


          
            if attack == "origin":
                    out_dict = self.network.forward(
                        template=self.z_dict1.tensors, search=x_dict.tensors, event_template=self.event_template.tensors,
                        event_search=event_search.tensors,  ce_template_mask=self.box_mask_z)
                    
                    # visual_for_attack(
                    #                 base_save_path="visual/ceosot/ori",
                    #                 senqice_name=senqice_name,
                    #                 ori_rgb_search=x_dict.tensors,
                    #                 adv=noise, 
                    #                 updated_rgb_adv_search=x_dict.tensors, 
                    #                 ori_event_search=event_search.tensors,
                    #                 updated_event_adv_search=event_search.tensors,
                    #                 frame_id = self.frame_id,
                    #                 current_time=self.current_time
                    #                 )

            elif attack == "attack_frame": #fgsm
                rgb_adv_search,event_adv_search = self.network.forward_attck_rgb_event_frame_sequce(
                                                    template=self.z_dict1.tensors, 
                                                    search=x_dict.tensors, 
                                                    event_template=self.event_template.tensors,                           
                                                    event_search=event_search.tensors,  
                                                    ce_template_mask=self.box_mask_z, 
                                                    adv_bbox=fack_bbox,
                                                    att_per = att_per,
                                                    cfg = self.cfg,
                                                    frame_id = self.frame_id,
                                                    iteration=10, 
                                                    current_time=self.current_time,
                                                    real_bbox=real_bbox_norm,
                                                    x_val_red_max=x_val_red_max, 
                                                    x_val_red_min=x_val_red_min,
                                                    x_val_green_max=x_val_green_max,
                                                    x_val_green_min=x_val_green_min,
                                                    x_val_blue_max=x_val_blue_max,
                                                    x_val_blue_min=x_val_blue_min,      
                                                    event_x_val_red_max=event_x_val_red_max,
                                                    event_x_val_red_min=event_x_val_red_min,
                                                    event_x_val_green_max=event_x_val_green_max,
                                                    event_x_val_green_min=event_x_val_green_min,
                                                    event_x_val_blue_max=event_x_val_blue_max,
                                                    event_x_val_blue_min=event_x_val_blue_min,
                                                    unit_pix = unit_pix,
                                                    gt_gaussian_maps = gt_gaussian_maps,
                                                    fack_gaussian_maps = fack_gaussian_maps
                                                    )

                out_dict = self.network.forward(
                                template=self.z_dict1.tensors, 
                                search=rgb_adv_search, 
                                event_template=self.event_template.tensors,
                                event_search=event_adv_search,  
                                ce_template_mask=self.box_mask_z)
                rgb_noise = rgb_adv_search - x_dict.tensors
                event_noise = event_adv_search - event_search.tensors
                rgb_noise[:,0,:,:] = torch.clamp(rgb_noise[:,0,:,:], max=x_val_red_max, min=x_val_red_min)
                rgb_noise[:,1,:,:] = torch.clamp(rgb_noise[:,1,:,:], max=x_val_green_max, min=x_val_green_min)
                rgb_noise[:,2,:,:] = torch.clamp(rgb_noise[:,2,:,:], max=x_val_blue_max, min=x_val_blue_min)    

                rgb_adv_search[:,0,:,:] = torch.clamp(rgb_adv_search[:,0,:,:], max=x_val_red_max, min=x_val_red_min)
                rgb_adv_search[:,1,:,:] = torch.clamp(rgb_adv_search[:,1,:,:], max=x_val_green_max, min=x_val_green_min)
                rgb_adv_search[:,2,:,:] = torch.clamp(rgb_adv_search[:,2,:,:], max=x_val_blue_max, min=x_val_blue_min)    

                event_adv_search[:,0,:,:] = torch.clamp(event_adv_search[:,0,:,:], max=event_x_val_red_max, min=event_x_val_red_min)
                event_adv_search[:,1,:,:] = torch.clamp(event_adv_search[:,1,:,:], max=event_x_val_green_max, min=event_x_val_green_min)
                event_adv_search[:,2,:,:] = torch.clamp(event_adv_search[:,2,:,:], max=event_x_val_blue_max, min=event_x_val_blue_min)

                # visual_for_attack(  
                #                     base_save_path="rgb",
                #                     senqice_name=senqice_name,
                #                     ori_search=x_dict.tensors,
                #                     adv = rgb_noise,
                #                     updated_adv_search =rgb_adv_search,
                #                     frame_id = self.frame_id ,
                #                     current_time = self.current_time,
                #                     )     
                # visual_for_attack(  
                #                     base_save_path="event",
                #                     senqice_name=senqice_name,
                #                     ori_search=event_search.tensors,
                #                     adv = event_noise,
                #                     updated_adv_search =event_adv_search,
                #                     frame_id = self.frame_id ,
                #                     current_time = self.current_time,
                #                     )     


        pred_score_map = out_dict['score_map'] #[1 1 16 16]
        response = self.output_window * pred_score_map
        pred_boxes = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'])
        pred_boxes = pred_boxes.view(-1, 4) #上面和原本的pre_boxes是一样的
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean( #改回正常的坐标
            dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result  先映射回原本的坐标 再限制范围裁剪  #改回正常的坐标
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        # for debug
        if self.debug:
            if not self.use_visdom:
                x1, y1, w, h = self.state
                image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.rectangle(image_BGR, (int(x1),int(y1)), (int(x1+w),int(y1+h)), color=(0,0,255), thickness=2)
                save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
                cv2.imwrite(save_path, image_BGR)
            else:
                self.visdom.register((image, info['gt_bbox'].tolist(), self.state), 'Tracking', 1, 'Tracking')
                self.visdom.register(torch.from_numpy(x_patch_arr).permute(2, 0, 1), 'image', 1, 'search_region')
                self.visdom.register(torch.from_numpy(self.z_patch_arr).permute(2, 0, 1), 'image', 1, 'template')
                self.visdom.register(pred_score_map.view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map')
                self.visdom.register((pred_score_map * self.output_window).view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map_hann')

                if 'removed_indexes_s' in out_dict and out_dict['removed_indexes_s']:
                    removed_indexes_s = out_dict['removed_indexes_s']
                    removed_indexes_s = [removed_indexes_s_i.cpu().numpy() for removed_indexes_s_i in removed_indexes_s]
                    masked_search = gen_visualization(x_patch_arr, removed_indexes_s)
                    self.visdom.register(torch.from_numpy(masked_search).permute(2, 0, 1), 'image', 1, 'masked_search')

                while self.pause_mode:
                    if self.step:
                        self.step = False
                        break

        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save},self.network.noise
        else:
            return {"target_bbox": self.state},self.network.noise

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)
    
    def add_hook(self):
        conv_features, enc_attn_weights, dec_attn_weights = [], [], []

        for i in range(12):
            self.network.backbone.blocks[i].attn.register_forward_hook(
                # lambda self, input, output: enc_attn_weights.append(output[1])
                lambda self, input, output: enc_attn_weights.append(output[1])
            )

        self.enc_attn_weights = enc_attn_weights


def get_tracker_class():
    return CEUTrack


def visual_for_attack(base_save_path,senqice_name,ori_search,adv, updated_adv_search, frame_id, current_time):

    time_folder_path = os.path.join(base_save_path, senqice_name)
    if not os.path.exists(time_folder_path):
        os.makedirs(time_folder_path)
    
    # 创建 frame_id 文件夹
    frame_folder_path = os.path.join(time_folder_path, f'frame_{frame_id}')
    if not os.path.exists(frame_folder_path):
        os.makedirs(frame_folder_path)
    

    # 归一化处理
    mean = torch.tensor([0.485, 0.456, 0.406]).view((1, 3, 1, 1)).cuda()
    std = torch.tensor([0.229, 0.224, 0.225]).view((1, 3, 1, 1)).cuda()
    ori_search = ori_search * std + mean
    adv = adv * std + mean
    updated_adv_search = updated_adv_search * std + mean
    
    # 将图像从 GPU 移动到 CPU 并转换为 numpy 数组
    ori_search = ori_search.squeeze(0).detach().cpu().numpy().transpose((1, 2, 0))  
    adv = adv.squeeze(0).detach().cpu().numpy().transpose((1, 2, 0))
    adv = np.clip(adv, 0, 1)
    updated_adv_search = updated_adv_search.squeeze(0).detach().cpu().numpy().transpose((1, 2, 0))
    
    # 单独保存每个图像
    plt.imsave(os.path.join(frame_folder_path, f'ori_search.png'), ori_search)
    plt.imsave(os.path.join(frame_folder_path, f'adv.png'), adv)
    plt.imsave(os.path.join(frame_folder_path, f'Current_adv.png'), updated_adv_search)