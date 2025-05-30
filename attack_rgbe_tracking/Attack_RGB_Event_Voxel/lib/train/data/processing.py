import copy

import torch
import torchvision.transforms as transforms
from lib.utils import TensorDict
import lib.train.data.processing_utils as prutils
import torch.nn.functional as F


def stack_tensors(x):
    if isinstance(x, (list, tuple)) and isinstance(x[0], torch.Tensor):
        return torch.stack(x)
    return x


class BaseProcessing:
    """ Base class for Processing. Processing class is used to process the data returned by a dataset, before passing it
     through the network. For example, it can be used to crop a search region around the object, apply various data
     augmentations, etc."""

    def __init__(self, transform=transforms.ToTensor(), template_transform=None, search_transform=None,
                 joint_transform=None):
        """
        args:
            transform       - The set of transformations to be applied on the images. Used only if template_transform or
                                search_transform is None.
            template_transform - The set of transformations to be applied on the template images. If None, the 'transform'
                                argument is used instead.
            search_transform  - The set of transformations to be applied on the search images. If None, the 'transform'
                                argument is used instead.
            joint_transform - The set of transformations to be applied 'jointly' on the template and search images.  For
                                example, it can be used to convert both template and search images to grayscale.
        """
        self.transform = {'template': transform if template_transform is None else template_transform,
                          'search': transform if search_transform is None else search_transform,
                          'joint': joint_transform}

    def __call__(self, data: TensorDict):
        raise NotImplementedError


class STARKProcessing(BaseProcessing):
    """ The processing class used for training LittleBoy. The images are processed in the following way.
    First, the target bounding box is jittered by adding some noise. Next, a square region (called search region )
    centered at the jittered target center, and of area search_area_factor^2 times the area of the jittered box is
    cropped from the image. The reason for jittering the target box is to avoid learning the bias that the target is
    always at the center of the search region. The search region is then resized to a fixed size given by the
    argument output_sz.
    """

    def __init__(self, search_area_factor, output_sz, center_jitter_factor, scale_jitter_factor,
                 mode='pair', settings=None, *args, **kwargs):
        """
        args:
            search_area_factor - The size of the search region  relative to the target size.
            output_sz - An integer, denoting the size to which the search region is resized. The search region is always
                        square.
            center_jitter_factor - A dict containing the amount of jittering to be applied to the target center before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            scale_jitter_factor - A dict containing the amount of jittering to be applied to the target size before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            mode - Either 'pair' or 'sequence'. If mode='sequence', then output has an extra dimension for frames
        """
        super().__init__(*args, **kwargs)
        self.search_area_factor = search_area_factor
        self.output_sz = output_sz
        self.center_jitter_factor = center_jitter_factor
        self.scale_jitter_factor = scale_jitter_factor
        self.mode = mode
        self.settings = settings

    def _get_jittered_box(self, box, mode):#数据增强 抖动跟踪狂 bbox tempalte/search
        """ Jitter the input box
        args:
            box - input bounding box
            mode - string 'template' or 'search' indicating template or search data

        returns:
            torch.Tensor - jittered box
        """
        #计算抖动大小
        jittered_size = box[2:4] * torch.exp(torch.randn(2) * self.scale_jitter_factor[mode])
        max_offset = (jittered_size.prod().sqrt() * torch.tensor(self.center_jitter_factor[mode]).float())
        jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (torch.rand(2) - 0.5)

        return torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0)

    def __call__(self, data: TensorDict):
        """
        args:
            data - The input data, should contain the following fields:
                'template_images', search_images', 'template_anno', 'search_anno'
        returns:
            TensorDict - output data block with following fields:
                'template_images', 'search_images', 'template_anno', 'search_anno', 'test_proposals', 'proposal_iou'
        """
        # Apply joint transforms
        if self.transform['joint'] is not None:
            data['template_images'], data['template_anno'], data['template_masks'] = self.transform['joint'](
                image=data['template_images'], bbox=data['template_anno'], mask=data['template_masks'])
            data['search_images'], data['search_anno'], data['search_masks'] = self.transform['joint'](
                image=data['search_images'], bbox=data['search_anno'], mask=data['search_masks'], new_roll=False)

        for s in ['template', 'search']:
            assert self.mode == 'sequence' or len(data[s + '_images']) == 1, \
                "In pair mode, num train/test frames must be 1"
            
            # Add a uniform noise to the center pos
            jittered_anno = [self._get_jittered_box(a, s) for a in data[s + '_anno']]
            # 所有边界框注释沿着新的维度 dim=0 堆叠起来 提取宽度 w 和高度 h
            # 2021.1.9 Check whether data is valid. Avoid too small bounding boxes
            w, h = torch.stack(jittered_anno, dim=0)[:, 2], torch.stack(jittered_anno, dim=0)[:, 3]
            #计算裁剪区域的大小
            crop_sz = torch.ceil(torch.sqrt(w * h) * self.search_area_factor[s])#2
            if (crop_sz < 1).any(): #太小不行
                data['valid'] = False
                # print("Too small box is found. Replace it with new data.")
                return data

            # Crop image region centered at jittered_anno box and get the attention mask
            crops, boxes, att_mask, mask_crops, crop_coor = prutils.jittered_center_crop(data[s + '_images'],
                                                                                         jittered_anno,
                                                                                         data[s + '_anno'],
                                                                                         self.search_area_factor[s],
                                                                                         self.output_sz[s],
                                                                                         masks=data[s + '_masks'])
            # Apply transforms
            data[s + '_images'], data[s + '_anno'], data[s + '_att'], data[s + '_masks'] = self.transform[s](
                image=crops, bbox=boxes, att=att_mask, mask=mask_crops, joint=False)
            # 这里是数据 取第一行数据
            data[s + '_event'] = torch.from_numpy(data[s + '_event'][0])
            z = copy.deepcopy(data[s + '_event'][:, 0]) #时间? 61行事件数据 每一列是特征或者坐标
            x, y = data[s + '_event'][:, 1], data[s + '_event'][:, 2]
            data[s + '_event'][:, 0] = x
            data[s + '_event'][:, 1] = y
            data[s + '_event'][:, 2] = z
            # crop to select voxels; template crop and search crop into the four times region.  // 10 resize
            x1, x2 = crop_coor[0][0] / 10, crop_coor[0][1] / 10 #裁剪区域 获取新坐标
            y1, y2 = crop_coor[0][2] / 10, crop_coor[0][3] / 10
            ### coor normalized to 0-1 becasue of box coor
            x_range, y_range = x2 - x1, y2 - y1  # 准换到裁剪区域的坐标并归一化处理 没有对时间归一化
            data[s + '_event'][:, 0] = (data[s + '_event'][:, 0]+0.5 - x1) / x_range
            data[s + '_event'][:, 1] = (data[s + '_event'][:, 1]+0.5 - y1) / y_range
            data[s + '_event'][:, 2] = (data[s + '_event'][:, 2]+0.5) / 19  
            indices = (data[s + '_event'][:, 0] >= 0) & (data[s + '_event'][:, 0] <= 1) & \
                      (data[s + '_event'][:, 1] >= 0) & (data[s + '_event'][:, 1] <= 1)
            data[s + '_event'] = torch.index_select(data[s + '_event'], dim=0, index=indices.nonzero().squeeze(1))
            # 上面是筛选归一化后的voxel在裁剪区域的坐标[61 19]->[2 19]
            # padding to 1024/4096 #插入维度 [2, 19] 变为 [1, 1, 61, 19]
            data[s + '_event'] = data[s + '_event'].unsqueeze(0).unsqueeze(0)
            if s in 'template' and (data[s + '_event'].shape[2] >= 1024):
                data[s + '_event'], _ = torch.topk(data[s + '_event'], k=1024, dim=2)
                pad_len = 0 #如果大于1024 不填充所以为0
            elif (s in 'template') and (data[s + '_event'].shape[2] < 1024):
                pad_len = 1024 - data[s + '_event'].shape[2]
            elif (s in 'search') and (data[s + '_event'].shape[2] < 4096):
                pad_len = 4096 - data[s + '_event'].shape[2]
            elif (s in 'search') and (data[s + '_event'].shape[2] >= 4096):
                data[s + '_event'], _ = torch.topk(data[s + '_event'], k=4096, dim=2)
                pad_len = 0
            else:
                print('the dataset is wrong.') #就是全填0
            data[s + '_event'] = F.pad(data[s + '_event'].transpose(-1, -2), (0, pad_len), mode='constant', value=0)

            for ele in data[s + '_att']:
                if (ele == 1).all():
                    data['valid'] = False
                    # print("Values of original attention mask are all one. Replace it with new data.")
                    return data
            # 2021.1.10 more strict conditions: require the donwsampled masks not to be all 1
            for ele in data[s + '_att']:
                feat_size = self.output_sz[s] // 16  # 16 is the backbone stride
                # (1,1,128,128) (1,1,256,256) --> (1,1,8,8) (1,1,16,16)
                mask_down = F.interpolate(ele[None, None].float(), size=feat_size).to(torch.bool)[0]
                if (mask_down == 1).all():
                    data['valid'] = False
                    # print("Values of down-sampled attention mask are all one. "
                    #       "Replace it with new data.")
                    return data

        data['valid'] = True
        # if we use copy-and-paste augmentation
        if data["template_masks"] is None or data["search_masks"] is None:
            data["template_masks"] = torch.zeros((1, self.output_sz["template"], self.output_sz["template"]))
            data["search_masks"] = torch.zeros((1, self.output_sz["search"], self.output_sz["search"]))
        # Prepare output
        if self.mode == 'sequence':
            data = data.apply(stack_tensors)
        else:
            data = data.apply(lambda x: x[0] if isinstance(x, list) else x)

        return data
