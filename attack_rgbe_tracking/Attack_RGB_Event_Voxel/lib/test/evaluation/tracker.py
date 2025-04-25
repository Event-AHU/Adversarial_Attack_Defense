import importlib
import os
from collections import OrderedDict
from lib.test.evaluation.environment import env_settings
import time
import cv2 as cv

from lib.utils.lmdb_utils import decode_img
from pathlib import Path
import numpy as np
import scipy.io as scio
import torch.nn.functional as F
import torch

def trackerlist(name: str, parameter_name: str, dataset_name: str, run_ids = None, display_name: str = None,
                result_only=False):
    """Generate list of trackers.
    args:
        name: Name of tracking method.
        parameter_name: Name of parameter file.
        run_ids: A single or list of run_ids.
        display_name: Name to be displayed in the result plots.
    """
    if run_ids is None or isinstance(run_ids, int):
        run_ids = [run_ids]
    return [Tracker(name, parameter_name, dataset_name, run_id, display_name, result_only) for run_id in run_ids]


class Tracker:
    """Wraps the tracker for evaluation and running purposes.
    args:
        name: Name of tracking method.
        parameter_name: Name of parameter file.
        run_id: The run id.
        display_name: Name to be displayed in the result plots.
    """

    def __init__(self, name: str, parameter_name: str, dataset_name: str, run_id: int = None, display_name: str = None,
                 result_only=False):
        assert run_id is None or isinstance(run_id, int)

        self.name = name
        self.parameter_name = parameter_name
        self.dataset_name = dataset_name
        self.run_id = run_id
        self.display_name = display_name

        env = env_settings()
        if self.run_id is None:
            self.results_dir = '{}/{}/{}'.format(env.results_path, self.name, self.parameter_name)
        else:
            self.results_dir = '{}/{}/{}_{:03d}'.format(env.results_path, self.name, self.parameter_name, self.run_id)
        if result_only:
            self.results_dir = '{}/{}'.format(env.results_path, self.name)

        tracker_module_abspath = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                              '..', 'tracker', '%s.py' % self.name))
        if os.path.isfile(tracker_module_abspath):
            tracker_module = importlib.import_module('lib.test.tracker.{}'.format(self.name))
            self.tracker_class = tracker_module.get_tracker_class()
        else:
            self.tracker_class = None

    def create_tracker(self, params):
        tracker = self.tracker_class(params, self.dataset_name)
        return tracker

    def run_sequence(self, seq, debug=None):
        """Run tracker on sequence.
        args:
            seq: Sequence to run the tracker on.
            visualization: Set visualization flag (None means default value specified in the parameters).
            debug: Set debug level (None means default value specified in the parameters).
            multiobj_mode: Which mode to use for multiple objects.
        """
        params = self.get_parameters()

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)

        params.debug = debug_

        # Get init information
        init_info = seq.init_info(0)
        # 初始化tracker
        tracker = self.create_tracker(params)
        # 处理数据 并运行
        output = self._track_sequence(tracker, seq, init_info)
        return output

    def _track_sequence(self, tracker, seq, init_info):
        # Define outputs
        # Each field in output is a list containing tracker prediction for each frame.

        # In case of single object tracking mode:
        # target_bbox[i] is the predicted bounding box for frame i
        # time[i] is the processing time for frame i

        # In case of multi object tracking mode:
        # target_bbox[i] is an OrderedDict, where target_bbox[i][obj_id] is the predicted box for target obj_id in
        # frame i
        # time[i] is either the processing time for frame i, or an OrderedDict containing processing times for each
        # object in frame i

        output = {'target_bbox': [],
                  'time': []}
        if tracker.params.save_all_boxes:
            output['all_boxes'] = []
            output['all_scores'] = []

        def _store_outputs(tracker_out: dict, defaults=None):
            defaults = {} if defaults is None else defaults
            for key in output.keys():
                val = tracker_out.get(key, defaults.get(key, None))
                if key in tracker_out or val is not None:
                    output[key].append(val)

        right_template = False  # flag for while
        i = 0  # template  index,
        # Initialize #初始化数据 最原始的数据 没有进行任何处理
        while not right_template:
            image = self._read_image(seq.frames[i])
            # event_image = self._read_image(seq.event_img_list[i])
            # image = cv.addWeighted(image, 1, event_image, 0.2, 0)

            # event_template  seq.event_frames[i] frame0000.mat' -1 1 0? 
            event_template = scio.loadmat(seq.event_frames[i])
            gt_bbox = seq.ground_truth_rect[i] #array([  9.7595, 114.2171,  75.1632, 106.1224])
            if (os.path.getsize(seq.event_frames[i]) == 0) or (np.isnan(event_template['features']).any()):
                event_template = torch.zeros([1024, 19]) # 如果为none 只能rgb跟踪
                right_template = True 
                print(seq.event_frames[i], i, 'template voxel is empty/nan, only based on rgb frames  for tracking.')
            elif gt_bbox[2]*gt_bbox[3] < 1:
                right_template = True #真值边界框的宽度和高度乘积是否小于 1，即边界框的面积是否过小。
                print(seq.event_frames[i], 'idx bbox zero, without any target or too small.')
                # i = i+1  #打印一条消息，说明边界框太小或没有目标。
            else:                                       #   event_template['coor'] [1393 3] 3 xyt     event_template['features'] [1393 16]
                event_template = np.concatenate((event_template['coor'], event_template['features']), axis=1)
                event_template = torch.from_numpy(event_template) # [1393 19]
                right_template = True
                init_info['init_bbox'] = gt_bbox
        
        start_time = time.time()
        out = tracker.initialize(image, event_template, init_info, idx=i)
        if out is None:
            out = {}
        prev_output = OrderedDict(out)
        if i != 0:
            for idx in range(0, i+1):
                target_box_template = seq.ground_truth_rect[idx]
                init_info = {'target_bbox': target_box_template, 'time': 0}
                _store_outputs(out, init_info)
        else:
            target_box_template = init_info.get('init_bbox')
            init_default = {'target_bbox': target_box_template,
                            'time': time.time() - start_time}
            _store_outputs(out, init_default)
        if tracker.params.save_all_boxes:
            init_default['all_boxes'] = out['all_boxes']
            init_default['all_scores'] = out['all_scores']
            _store_outputs(out, init_default)
        # search区域 0是模型 1是搜索区域 所以search从1开始
        for frame_num, (frame_path, frame_event_path) in enumerate(zip(seq.frames[i+1:], seq.event_frames[i+1:]), start=1):
        # for frame_num, (frame_path, frame_event_path, event_img_path) in enumerate(zip(seq.frames[i+1:], seq.event_frames[i+1:], seq.event_img_list[i+1:]), start=1):
            image = self._read_image(frame_path)
            # event_img = self._read_image(event_img_path)
            # image = cv.addWeighted(image, 1, event_img, 0.2, 0)
            grandparent_dir = os.path.dirname(os.path.dirname(frame_path))
            senqice_name = os.path.basename(grandparent_dir)
            # event_search
            if os.path.getsize(frame_event_path) == 0:
                event_search = torch.zeros([4096, 19])
                print(frame_event_path, 'idx frame of search voxel is empty. only based rgb')
            else: #走这里
                event_search = scio.loadmat(frame_event_path)
                event_search = np.concatenate((event_search['coor'], event_search['features']), axis=1)
                event_search = torch.from_numpy(event_search)
                if np.isnan(event_search).any():
                    event_search = torch.zeros([4096, 19])
                    print(frame_event_path, 'idx frame of search voxel exist  nan, only based on rgb.')

            start_time = time.time()

            info = seq.frame_info(frame_num) #获取当前帧的信息
            info['previous_output'] = prev_output #将前一帧的输出结果存储到 info 中

            if len(seq.ground_truth_rect) > 1: #获取真实边界框
                info['gt_bbox'] = seq.ground_truth_rect[frame_num]
                # 将 info['gt_bbox'] 转换为形状为 [1, 4] 的张量
                real_box = torch.tensor(info['gt_bbox']).unsqueeze(0)  # Shape: [1, 4]
                # 将张量移动到 CUDA 设备上
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                real_bbox = real_box.to(device)
            
            if frame_num == 0 or frame_num == 1:
                att_per = 0
                att_per_rgb = 0
                att_per_event =0 
                att_x = 0
                att_y = 0
                att_t = 0
                att_double_rgb = 0
                print('frame_num:', frame_num)
            
            # if frame_num % 1 == 1:
            #     att_per = 0 
            #     print('frame_num:', frame_num)
            tracker.k = 1
            out,att_per,att_per_rgb,att_per_event,att_x,att_y,att_t = tracker.track(image, event_search,info=info,
                                        att_per=att_per,
                                        att_per_rgb = att_per_rgb,
                                        att_per_event=att_per_event,
                                        att_x = att_x,
                                        att_y = att_y,
                                        att_t = att_t,
                                        att_double_rgb = att_double_rgb,
                                        real_bbox=real_bbox,
                                        senqice_name=senqice_name)
            prev_output = OrderedDict(out)
            _store_outputs(out, {'time': time.time() - start_time})

        for key in ['target_bbox', 'all_boxes', 'all_scores']:
            if key in output and len(output[key]) <= 1:
                output.pop(key)

        return output

    def run_video(self, videofilepath, optional_box=None, debug=None, visdom_info=None, save_results=False):
        """Run the tracker with the vieofile.
        args:
            debug: Debug level.
        """

        params = self.get_parameters()

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)
        params.debug = debug_

        params.tracker_name = self.name
        params.param_name = self.parameter_name
        # self._init_visdom(visdom_info, debug_)

        multiobj_mode = getattr(params, 'multiobj_mode', getattr(self.tracker_class, 'multiobj_mode', 'default'))

        if multiobj_mode == 'default':
            tracker = self.create_tracker(params)

        elif multiobj_mode == 'parallel':
            tracker = MultiObjectWrapper(self.tracker_class, params, self.visdom, fast_load=True)
        else:
            raise ValueError('Unknown multi object mode {}'.format(multiobj_mode))

        assert os.path.isfile(videofilepath), "Invalid param {}".format(videofilepath)
        ", videofilepath must be a valid videofile"

        output_boxes = []

        cap = cv.VideoCapture(videofilepath)
        display_name = 'Display: ' + tracker.params.tracker_name
        cv.namedWindow(display_name, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
        cv.resizeWindow(display_name, 960, 720)
        success, frame = cap.read()
        cv.imshow(display_name, frame)

        def _build_init_info(box):
            return {'init_bbox': box}

        if success is not True:
            print("Read frame from {} failed.".format(videofilepath))
            exit(-1)
        if optional_box is not None:
            assert isinstance(optional_box, (list, tuple))
            assert len(optional_box) == 4, "valid box's foramt is [x,y,w,h]"
            tracker.initialize(frame, _build_init_info(optional_box))
            output_boxes.append(optional_box)
        else:
            while True:
                # cv.waitKey()
                frame_disp = frame.copy()

                cv.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL,
                           1.5, (0, 0, 0), 1)

                x, y, w, h = cv.selectROI(display_name, frame_disp, fromCenter=False)
                init_state = [x, y, w, h]
                tracker.initialize(frame, _build_init_info(init_state))
                output_boxes.append(init_state)
                break

        while True:
            ret, frame = cap.read()

            if frame is None:
                break

            frame_disp = frame.copy()

            # Draw box
            out = tracker.track(frame)
            state = [int(s) for s in out['target_bbox']]
            output_boxes.append(state)

            cv.rectangle(frame_disp, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]),
                         (0, 255, 0), 5)

            font_color = (0, 0, 0)
            cv.putText(frame_disp, 'Tracking!', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)
            cv.putText(frame_disp, 'Press r to reset', (20, 55), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)
            cv.putText(frame_disp, 'Press q to quit', (20, 80), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)

            # Display the resulting frame
            cv.imshow(display_name, frame_disp)
            key = cv.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('r'):
                ret, frame = cap.read()
                frame_disp = frame.copy()

                cv.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                           (0, 0, 0), 1)

                cv.imshow(display_name, frame_disp)
                x, y, w, h = cv.selectROI(display_name, frame_disp, fromCenter=False)
                init_state = [x, y, w, h]
                tracker.initialize(frame, _build_init_info(init_state))
                output_boxes.append(init_state)

        # When everything done, release the capture
        cap.release()
        cv.destroyAllWindows()

        if save_results:
            if not os.path.exists(self.results_dir):
                os.makedirs(self.results_dir)
            video_name = Path(videofilepath).stem
            base_results_path = os.path.join(self.results_dir, 'video_{}'.format(video_name))

            tracked_bb = np.array(output_boxes).astype(int)
            bbox_file = '{}.txt'.format(base_results_path)
            np.savetxt(bbox_file, tracked_bb, delimiter='\t', fmt='%d')


    def get_parameters(self):
        """Get parameters."""
        param_module = importlib.import_module('lib.test.parameter.{}'.format(self.name))
        params = param_module.parameters(self.parameter_name)
        return params

    def _read_image(self, image_file: str):
        if isinstance(image_file, str): #为什么这里的是bmp 可是文件里面是png
            im = cv.imread(image_file) # [0 255]
            return cv.cvtColor(im, cv.COLOR_BGR2RGB) #交换颜色通道 BGR->RGB
        elif isinstance(image_file, list) and len(image_file) == 2:
            return decode_img(image_file[0], image_file[1])
        else:
            raise ValueError("type of image_file should be str or list")



