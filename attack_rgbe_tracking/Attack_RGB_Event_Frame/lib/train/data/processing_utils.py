import torch
import math
import cv2 as cv
import torch.nn.functional as F
import numpy as np

'''modified from the original test implementation
Replace cv.BORDER_REPLICATE with cv.BORDER_CONSTANT
Add a variable called att_mask for computing attention and positional encoding later'''


def sample_target(im, target_bb, search_area_factor, output_sz=None, mask=None):
    """ Extracts a square crop centered at target_bb box, of area search_area_factor^2 times target_bb area

    args:
        im - cv image
        target_bb - target box [x, y, w, h] 基于bbox裁剪模版区域
        search_area_factor - Ratio of crop size to target size
        output_sz - (float) Size to which the extracted crop is resized (always square). If None, no resizing is done.

    returns:
        cv image - extracted crop
        float - the factor by which the crop has been resized to make the crop size equal output_size
    """ #269 346 3
    if not isinstance(target_bb, list):
        x, y, w, h = target_bb.tolist()
    else:
        x, y, w, h = target_bb # 11 113 78 106
    # Crop image 计算裁剪区域的大小 366
    crop_sz = math.ceil(math.sqrt(w * h) * search_area_factor)
    # 处理过小的边界框
    if crop_sz < 1:
        w = h = 1
        crop_sz = 1
    #     raise Exception('Too small bounding box.')
    #计算裁剪区域的坐标：x 是bbox的坐标
    x1 = round(x + 0.5 * w - crop_sz * 0.5) #-132
    x2 = x1 + crop_sz #234

    y1 = round(y + 0.5 * h - crop_sz * 0.5) #-16
    y2 = y1 + crop_sz
    #计算需要填充的像素数：
    x1_pad = max(0, -x1) #坐标需要填充的像素数
    x2_pad = max(x2 - im.shape[1] + 1, 0) #右边需要填充的像素数 负数说明需要填充的像素还在图片内 所以不需要填充 变为0 

    y1_pad = max(0, -y1) #上面需要填充的像素
    y2_pad = max(y2 - im.shape[0] + 1, 0)#91 下面需要填充的像素

    # Crop target   
    # 原始的图像坐标 未填充的 即我们需要裁剪的区域
    crop_coor = [x1 + x1_pad, x2 - x2_pad, y1 + y1_pad, y2 - y2_pad] # 0 234 0 259
     # y是行索引 x是列索引 : 是保留所有通道
    im_crop = im[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad, :] 

    if mask is not None:
        mask_crop = mask[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad]

    # Pad [366 366 3]
    # 填充 cv.copyMakeBorder(src, top, bottom, left, right, borderType[, dst[, value]]) → dst
    im_crop_padded = cv.copyMakeBorder(im_crop, y1_pad, y2_pad, x1_pad, x2_pad, cv.BORDER_CONSTANT)
    # deal with attention mask
    if crop_sz == 0: # 裁剪区域很小
        im_crop_padded = cv.resize(im, (output_sz, output_sz))
        im_mask = np.zeros_like(im)
        att_mask = cv.resize(im_mask, (output_sz, output_sz)).astype(np.bool_)
        return im_crop_padded, 1, att_mask, crop_coor
    # 179 179 生成注意掩码 标记哪些区域是有效的 哪些区域是填充无效的
    H, W, _ = im_crop_padded.shape
    att_mask = np.ones((H, W))
    end_x, end_y = -x2_pad, -y2_pad # 0 -91
    if y2_pad == 0: #如果某个方向没有填充（即 x2_pad 或 y2_pad 为0），则将对应的结束坐标设为 None，这样在后续的切片操作中会取到数组的末尾。
        end_y = None
    if x2_pad == 0:
        end_x = None #实际包含图像的部分（即未被填充的部分）设置为0，表示这些区域是有效的
    att_mask[y1_pad:end_y, x1_pad:end_x] = 0
    if mask is not None: # 如果提供了掩码 mask，则对裁剪区域的掩码进行相同的填充处理
        mask_crop_padded = F.pad(mask_crop, pad=(x1_pad, x2_pad, y1_pad, y2_pad), mode='constant', value=0)

    if output_sz is not None:
        resize_factor = output_sz / crop_sz # 256 / 366
        # 366 直接resize成256
        im_crop_padded = cv.resize(im_crop_padded, (output_sz, output_sz)) #resize成输出尺寸
        att_mask = cv.resize(att_mask, (output_sz, output_sz)).astype(np.bool_) #同样将注意力掩码调整到指定的输出大小，并将其类型转换为布尔型
        if mask is None: #返回
            return im_crop_padded, resize_factor, att_mask, crop_coor
        mask_crop_padded = \
        F.interpolate(mask_crop_padded[None, None], (output_sz, output_sz), mode='bilinear', align_corners=False)[0, 0]
        return im_crop_padded, resize_factor, att_mask, mask_crop_padded, crop_coor

    else:
        if mask is None:
            return im_crop_padded, att_mask.astype(np.bool_), 1.0
        return im_crop_padded, 1.0, att_mask.astype(np.bool_), mask_crop_padded, crop_coor

#原始图像中的边界框坐标转换到裁剪图像中的坐标
def transform_image_to_crop(box_in: torch.Tensor, box_extract: torch.Tensor, resize_factor: float,
                            crop_sz: torch.Tensor, normalize=False) -> torch.Tensor:
    """ Transform the box co-ordinates from the original image co-ordinates to the co-ordinates of the cropped image
    args:
        box_in - the box for which the co-ordinates are to be transformed
        box_extract - the box about which the image crop has been extracted.
        resize_factor - the ratio between the original image scale and the scale of the image crop
        crop_sz - size of the cropped image

    returns:
        torch.Tensor - transformed co-ordinates of box_in
    """
    box_extract_center = box_extract[0:2] + 0.5 * box_extract[2:4]

    box_in_center = box_in[0:2] + 0.5 * box_in[2:4]

    box_out_center = (crop_sz - 1) / 2 + (box_in_center - box_extract_center) * resize_factor
    box_out_wh = box_in[2:4] * resize_factor

    box_out = torch.cat((box_out_center - 0.5 * box_out_wh, box_out_wh))
    if normalize:
        return box_out / crop_sz[0] # 归一化输入框 128
    else:
        return box_out
    # box_out_normalized = [36.6608, 25.5558, 53.6784, 75.8884] / 128
    # box_out_normalized = [0.2864, 0.2004, 0.4194, 0.5937]

def jittered_center_crop(frames, box_extract, box_gt, search_area_factor, output_sz, masks=None):
    """ For each frame in frames, extracts a square crop centered at box_extract, of area search_area_factor^2
    times box_extract area. The extracted crops are then resized to output_sz. Further, the co-ordinates of the box
    box_gt are transformed to the image crop co-ordinates

    args:
        frames - list of frames
        box_extract - list of boxes of same length as frames. The crops are extracted using anno_extract
        box_gt - list of boxes of same length as frames. The co-ordinates of these boxes are transformed from
                    image co-ordinates to the crop co-ordinates
        search_area_factor - The area of the extracted crop is search_area_factor^2 times box_extract area
        output_sz - The size to which the extracted crops are resized

    returns:
        list - list of image crops
        list - box_gt location in the crop co-ordinates
        """

    if masks is None:
        crops_resize_factors = [sample_target(f, a, search_area_factor, output_sz)
                                for f, a in zip(frames, box_extract)]
        frames_crop, resize_factors, att_mask = zip(*crops_resize_factors)
        masks_crop = None
    else:#走这里 裁剪搜索区域
        crops_resize_factors = [sample_target(f, a, search_area_factor, output_sz, m)
                                for f, a, m in zip(frames, box_extract, masks)]
        frames_crop, resize_factors, att_mask, masks_crop, crop_coor = zip(*crops_resize_factors)
    # frames_crop: tuple of ndarray (128,128,3), att_mask: tuple of ndarray (128,128)
    crop_sz = torch.Tensor([output_sz, output_sz])

    # find the bb location in the crop
    '''Note that here we use normalized coord'''
    box_crop = [transform_image_to_crop(a_gt, a_ex, rf, crop_sz, normalize=True)
                for a_gt, a_ex, rf in zip(box_gt, box_extract, resize_factors)]  # (x1,y1,w,h) list of tensors

    return frames_crop, box_crop, att_mask, masks_crop, crop_coor


def transform_box_to_crop(box: torch.Tensor, crop_box: torch.Tensor, crop_sz: torch.Tensor, normalize=False) -> torch.Tensor:
    """ Transform the box co-ordinates from the original image co-ordinates to the co-ordinates of the cropped image
    args:
        box - the box for which the co-ordinates are to be transformed
        crop_box - bounding box defining the crop in the original image
        crop_sz - size of the cropped image

    returns:
        torch.Tensor - transformed co-ordinates of box_in
    """

    box_out = box.clone()
    box_out[:2] -= crop_box[:2]

    scale_factor = crop_sz / crop_box[2:]

    box_out[:2] *= scale_factor
    box_out[2:] *= scale_factor
    if normalize:
        return box_out / crop_sz[0]
    else:
        return box_out

