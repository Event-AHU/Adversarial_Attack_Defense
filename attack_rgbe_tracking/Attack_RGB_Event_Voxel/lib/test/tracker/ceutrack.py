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
from torch.nn.functional import gumbel_softmax
from lib.gumbel_softmax.tracking_gumbel.HardDiffArgmax1 import HardDiffArgmax
from scripts.show_CAM import getCAM

class CEUTrack(BaseTracker):
    def __init__(self, params, dataset_name):
        super(CEUTrack, self).__init__(params)
        network = build_ceutrack(params.cfg, training=False)
        # state_dict = torch.load(self.params.checkpoint)
        # missing_keys = ["gumbel_alpha"]
        # for key in missing_keys:
        #     if key in state_dict:
        #         del state_dict[key]
        # network.load_state_dict(state_dict, strict=True)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=False)
        print('load model from {}'.format(self.params.checkpoint))
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor() #处理数据
        self.state = None
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

    def initialize(self, image, event_template,  info: dict, idx=0):
        # forward the template once # 模版图像 128 *128 缩放大小 有效区域 模版原始坐标
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
        event_template = event_template.cuda() #[100, 19] 100个voxel 肯定是voxel 因为维度是19个维度
        z = copy.deepcopy(event_template[:, 0]) #第一个维度是时间？
        x, y = event_template[:, 1], event_template[:, 2]
        event_template[:, 0] = x
        event_template[:, 1] = y
        event_template[:, 2] = z #为啥除以10 [100 19]
        x1, x2 = crop_coor[0] / 10, crop_coor[1] / 10 #crop_coor 最原始图像的WH [0, 345, 0, 259]
        y1, y2 = crop_coor[2] / 10, crop_coor[3] / 10
        x_range, y_range = x2-x1, y2-y1 #坐标归一化
        event_template[:, 0] = (event_template[:, 0]+0.5 - x1) / x_range # 0 1 2分别代表xyt 1393组数据
        event_template[:, 1] = (event_template[:, 1]+0.5 - y1) / y_range
        event_template[:, 2] = (event_template[:, 2]+0.5) / 19
        indices = (event_template[:, 0] >= 0) & (event_template[:, 0] <= 1) & (
                event_template[:, 1] >= 0) & (event_template[:, 1] <= 1) #xy有效的区域
        event_template = torch.index_select(event_template, dim=0, index=indices.nonzero().squeeze(1))
        # [1393 19] -> [1 1 1063 19] 上部被筛选掉了
        event_template = event_template.unsqueeze(0).unsqueeze(0)
        if event_template.shape[2] >= 1024: # 大于1024 选前1024个
            event_template, _ = torch.topk(event_template, k=1024, dim=2)
            pad_len_temp = 0
        elif event_template.shape[2] < 1024:
            pad_len_temp = 1024 - event_template.shape[2] # 填充voxel数量 [1 1 100 19] [1 1 19 1024]
        event_template = F.pad(event_template.transpose(-1, -2), (0, pad_len_temp), mode='constant', value=0)
        self.event_template = event_template #[-1 -2]交换位置
        # save states
        self.state = info['init_bbox'] #边界框
        self.frame_id = idx #0
        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}
    #跟一帧
    def track(self, image, event_search,att_per,att_per_rgb,
                    att_per_event,real_bbox, 
                    att_double_rgb,
                    att_x,att_y,att_t,
                    senqice_name,
                    info: dict = None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        H, W, _ = image.shape #260 346
        self.frame_id += 1
        x_patch_arr, resize_factor, x_amask_arr, crop_coor = sample_target(image, self.state, self.params.search_factor, # 4
                                                                output_sz=self.params.search_size)   # (x1, y1, w, h)
        output_sz=self.params.search_size
        crop_sz = torch.Tensor([output_sz, output_sz]).to(device)
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

        
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)
        attack = self.cfg.ATX.attack
        # origin/attack_voxel

        if attack =="origin":
            event_search = event_search.cuda()
            z = copy.deepcopy(event_search[:, 0])
            x, y = event_search[:, 1], event_search[:, 2]
            event_search[:, 0] = x
            event_search[:, 1] = y
            event_search[:, 2] = z
            # 像素空间坐标转化为voxel空间坐标
            x1, x2 = crop_coor[0] / 10, crop_coor[1] / 10
            y1, y2 = crop_coor[2] / 10, crop_coor[3] / 10
            x_range, y_range = x2-x1, y2-y1
            # +0.5 中心坐标转为voxel中心坐标
            event_search[:, 0] = (event_search[:, 0]+0.5 - x1) / x_range   # x voxel center
            event_search[:, 1] = (event_search[:, 1]+0.5 - y1) / y_range   # y voxel center
            event_search[:, 2] = (event_search[:, 2]+0.5) / 19                     # z voxel center (times)        
            indices = (event_search[:, 0] >= 0) & (event_search[:, 0] <= 1) & (
                    event_search[:, 1] >= 0) & (event_search[:, 1] <= 1)
            event_search = torch.index_select(event_search, dim=0, index=indices.nonzero().squeeze(1))

            event_search = event_search.unsqueeze(0).unsqueeze(0)
            # event frame  need to keep same length  16,12-->20, 20
            if event_search.shape[2] < 4096:
                pad_len_search = 4096 - event_search.shape[2]
            else:
                event_search, _ = torch.topk(event_search, k=4096, dim=2)
                pad_len_search = 0

            event_search = F.pad(event_search.transpose(-1, -2), (0, pad_len_search), mode='constant', value=0)
            unit_coordinate_x = 0.1/x_range
            unit_coordinate_y = 0.1/y_range 
            unit_coordinate_t = 1/19    
        elif attack == "attack_voxel":
            event_search = event_search.cuda()
            z = copy.deepcopy(event_search[:, 0])
            x, y = event_search[:, 1], event_search[:, 2]
            event_search[:, 0] = x
            event_search[:, 1] = y
            event_search[:, 2] = z            
            # 像素空间坐标转化为voxel空间坐标 坐标系转换
            x1, x2 = crop_coor[0] / 10, crop_coor[1] / 10
            y1, y2 = crop_coor[2] / 10, crop_coor[3] / 10
            x_range, y_range = x2-x1, y2-y1
            
            # +0.5 中心坐标转为voxel中心坐标 正则化坐标
            event_search[:, 0] = (event_search[:, 0]+0.5 - x1) / x_range   # x voxel center
            event_search[:, 1] = (event_search[:, 1]+0.5 - y1) / y_range   # y voxel center
            event_search[:, 2] = (event_search[:, 2]+0.5) / 19                     # z voxel center (times)        
                       
            # 不在x_range和y_range范围内的点去掉 归一化后的点只会保留在[0 1]的点
            indices = (event_search[:, 0] >= 0) & (event_search[:, 0] <= 1) & (
                    event_search[:, 1] >= 0) & (event_search[:, 1] <= 1)
            event_search = torch.index_select(event_search, dim=0, index=indices.nonzero().squeeze(1))

            event_search = event_search.unsqueeze(0).unsqueeze(0)
            # event frame  need to keep same length  16,12-->20, 20
            if event_search.shape[2] < 4096:
                pad_len_search = 4096 - event_search.shape[2]
            else:
                # 第二个维度选取最大值？
                event_search, _ = torch.topk(event_search, k=4096, dim=2)
                pad_len_search = 0
                event_search = event_search.transpose(-1, -2)
            # 每一帧都会随机生成 空事件扰动不具备传递性
                            
            unit_coordinate_x = 0.1/x_range
            unit_coordinate_y = 0.1/y_range 
            unit_coordinate_t = 1/19

            if pad_len_search > 0:
                null_event = []
                target_x1 = self.cfg.ATX.X
                target_y1 = self.cfg.ATX.Y
                target_x2 = target_x1 + rate_wd1
                target_y2 = target_y1 + rate_wd2
                # 转换为voxel空间
                target_x1, target_x2 = target_x1 / 10, target_x2 / 10
                target_y1, target_y2 = target_y1 / 10, target_y2 / 10
                
                # +0.5 中心坐标转为voxel中心坐标 归一化后的点只会保留在[0 1]的点
                target_x1 = (target_x1+0.5 - x1) / x_range   # x voxel center
                target_x2 = (target_x2+0.5 - x1) / x_range   # x voxel center
                target_y1 = (target_y1+0.5 - y1) / y_range   # y voxel center
                target_y2 = (target_y2+0.5 - y1) / y_range   # y voxel center

                x_coords = torch.arange(target_x1, target_x2 + unit_coordinate_x, unit_coordinate_x)
                y_coords = torch.arange(target_y1, target_y2 + unit_coordinate_y, unit_coordinate_y)

                target_coordinates_or = torch.tensor([[x, y] for x in x_coords for y in y_coords]).to(device)
                gap = 0.0526 #0.0526 单位时间
                if len(target_coordinates_or) >= pad_len_search:
                    random_indices = random.sample(range(len(target_coordinates_or)), pad_len_search)
                    selected_coordinates = target_coordinates_or[random_indices]
                    target_coordinates = selected_coordinates
                    null_event = torch.cat([target_coordinates, torch.zeros(target_coordinates.size(0), 17).to(device)], dim=1)
                    null_event[len(null_event)-len(selected_coordinates):,2] = gap
                else:
                    null_event_len = len(target_coordinates_or) #672
                    copy_null_event = int(pad_len_search / null_event_len) 
                    target_coordinates_or = target_coordinates_or.repeat(copy_null_event, 1)
                    pad_null_event = pad_len_search - len(target_coordinates_or)
                    random_indices = random.sample(range(len(target_coordinates_or)), pad_null_event)
                    selected_coordinates = target_coordinates_or[random_indices]
                    target_coordinates_or = torch.cat([target_coordinates_or,selected_coordinates],dim=0)
                    null_event = torch.cat([target_coordinates_or, torch.zeros(target_coordinates_or.size(0), 17).to(device)], dim=1)
                    begin = null_event_len
                    end =  null_event_len * 2 
                    for i in range(copy_null_event):
                        null_event[begin:end, 2] = gap
                        begin += null_event_len
                        end += null_event_len
                        gap += 0.001
                    null_event[len(null_event)-len(selected_coordinates):,2] = gap

                null_event = null_event.unsqueeze(0).unsqueeze(0)
                event_search = torch.cat([event_search, null_event], dim=2).transpose(-1, -2)

        with torch.no_grad():
            x_dict = search
            ones_tensor = torch.ones_like(x_dict.tensors)
            zero_tensor = torch.zeros_like(x_dict.tensors)
            mean = torch.tensor([0.485, 0.456, 0.406]).view((1, 3, 1, 1)).cuda()
            std = torch.tensor([0.229, 0.224, 0.225]).view((1, 3, 1, 1)).cuda()
            ones_tensor = ((ones_tensor / 255.0) - mean) / std 
            zero_tensor = ((zero_tensor / 255.0) - mean) / std 
            unit_pix = (ones_tensor-zero_tensor).to(device)
            unit_time = 1/19

            x_val_red_max = torch.max(x_dict.tensors[:,0,:,:]).to(device)
            x_val_red_min = torch.min(x_dict.tensors[:,0,:,:]).to(device)
            x_val_green_max = torch.max(x_dict.tensors[:,1,:,:]).to(device)
            x_val_green_min = torch.min(x_dict.tensors[:,1,:,:]).to(device)
            x_val_blue_max = torch.max(x_dict.tensors[:,2,:,:]).to(device)
            x_val_blue_min = torch.min(x_dict.tensors[:,2,:,:]).to(device)

            if attack == "origin":
                out_dict = self.network.forward(
                    template=self.z_dict1.tensors, search=x_dict.tensors, event_template=self.event_template,
                    event_search=event_search,  ce_template_mask=self.box_mask_z)
                # noise = torch.randn(1, 3, 256, 256).to(device)

                # visual_for_attack( 
                #                 base_save_path = "/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c01/DATA/chenqiang/AttackTracking/CEUTrack/visual/ceosot/ori" ,
                #                 ori_search=x_dict.tensors,
                #                 adv = noise,
                #                 updated_adv_search = x_dict.tensors,
                #                 frame_id = self.frame_id ,
                #                 current_time = self.current_time,
                #                 )
            elif attack == "attack_voxel":
                    rgb_adv_search = self.network.forward_attck_img(template=self.z_dict1.tensors, 
                                                    ori_search = x_dict.tensors,          
                                                    search=x_dict.tensors, 
                                                    event_template=self.event_template,                           
                                                    event_search=event_search,  
                                                    ce_template_mask=self.box_mask_z, 
                                                    adv_bbox=fack_bbox,att_per = att_per,
                                                    cfg = self.cfg,
                                                    frame_id = self.frame_id,
                                                    current_time=self.current_time,
                                                    real_bbox=real_bbox_norm,
                                                    iteration=1,
                                                    x_val_red_max=x_val_red_max, 
                                                    x_val_red_min=x_val_red_min,
                                                    x_val_green_max=x_val_green_max,
                                                    x_val_green_min=x_val_green_min,
                                                    x_val_blue_max=x_val_blue_max,
                                                    x_val_blue_min=x_val_blue_min,  
                                                    unit_pix = unit_pix,
                                                    gt_gaussian_maps = gt_gaussian_maps,
                                                    fack_gaussian_maps = fack_gaussian_maps)
                    att_per = rgb_adv_search - x_dict.tensors


                    rgb_adv_search[:,0,:,:] = torch.clamp(rgb_adv_search[:,0,:,:], max=x_val_red_max, min=x_val_red_min)
                    rgb_adv_search[:,1,:,:] = torch.clamp(rgb_adv_search[:,1,:,:], max=x_val_green_max, min=x_val_green_min)
                    rgb_adv_search[:,2,:,:] = torch.clamp(rgb_adv_search[:,2,:,:], max=x_val_blue_max, min=x_val_blue_min)

                    time = event_search[:,:,2,:]
                    coor_x = event_search[:,:,0,:]
                    coor_y = event_search[:,:,1,:]

                    max_value = torch.max(time)
                    min_value = torch.min(time)

                    event_adv_search,time_adv,x_adv,y_adv = self.network.forward_null_event_shift_xyt(
                            template=self.z_dict1.tensors, 
                            ori_search = x_dict.tensors,
                            search=rgb_adv_search, 
                            event_template=self.event_template,
                            event_search=event_search,
                            ce_template_mask=self.box_mask_z, 
                            adv_bbox=fack_bbox,
                            att_t = att_t,
                            att_x=att_x,
                            att_y=att_y,
                            unit_time = unit_coordinate_t,
                            unit_x = unit_coordinate_x,
                            unit_y = unit_coordinate_y,
                            iteration=1,
                            cfg = self.cfg,
                            frame_id = self.frame_id,
                            current_time=self.current_time,
                            real_bbox=real_bbox_norm,
                            time_val_min=min_value,
                            time_val_max=max_value,
                            gt_gaussian_maps = gt_gaussian_maps,
                            fack_gaussian_maps = fack_gaussian_maps,
                            )#null_event=null_event
                
                    att_t = time_adv - time #x_crop 原始图 更新噪声
                    att_x = x_adv - coor_x #x_crop 原始图 更新噪声
                    att_y = y_adv - coor_y #x_crop 原始图 更新噪声


                    out_dict = self.network.forward(
                        template=self.z_dict1.tensors, search=rgb_adv_search, 
                        event_template=self.event_template,
                        event_search=event_adv_search,  
                        ce_template_mask=self.box_mask_z)   
                    # visual_for_attack( 
                    #     base_save_path = "/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c01/DATA/chenqiang/AttackTracking/CEUTrack/visual/ceosot/fgsm" ,
                    #     ori_search=x_dict.tensors,
                    #     adv = att_per,
                    #     updated_adv_search = rgb_adv_search,
                    #     frame_id = self.frame_id ,
                    #     current_time = self.current_time,
                    #                 )

        # add hann windows
        pred_score_map = out_dict['score_map'] #[1 1 16 16]
        response = self.output_window * pred_score_map
        pred_boxes = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'])
        
        pred_boxes = pred_boxes.view(-1, 4) #上面和原本的pre_boxes是一样的
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean( #改回正常的坐标
            dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result  先映射回原本的坐标 再限制范围裁剪  #改回正常的坐标
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)
        # getCAM(response, x_patch_arr, self.frame_id,senqice_name)
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
                    "all_boxes": all_boxes_save},att_per,att_per_rgb,att_per_event,att_x,att_y,att_t
        else:
            return {"target_bbox": self.state},att_per,att_per_rgb,att_per_event,att_x,att_y,att_t #返回

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

def visual_for_attack(base_save_path,ori_search,adv, updated_adv_search, frame_id, current_time):

    time_folder_path = os.path.join(base_save_path, current_time)
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



