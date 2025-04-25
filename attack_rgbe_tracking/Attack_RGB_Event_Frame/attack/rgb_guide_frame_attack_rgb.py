import os
import sys
import argparse

import time
from datetime import timedelta

import numpy as np
import multiprocessing
import os
import sys
from itertools import product
from collections import OrderedDict
from event_CEUTrack.lib.test.evaluation import Sequence, Tracker
import torch
prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

from event_CEUTrack.lib.test.evaluation import get_dataset
from event_CEUTrack.lib.test.evaluation.running import run_dataset
from event_CEUTrack.lib.test.evaluation.tracker import Tracker
from event_CEUTrack.lib.test.evaluation.running import _save_tracker_output

def run_tracker(tracker_name, tracker_param, run_id=None, dataset_name='otb', 
                sequence=None, debug=0, threads=0,
                num_gpus=8):
    """Run tracker on sequence or dataset.
    args:
        tracker_name: Name of tracking method.
        tracker_param: Name of parameter file.
        run_id: The run id.
        dataset_name: Name of dataset
        sequence: Sequence number or name.
        debug: Debug level.
        threads: Number of threads.
    """

    dataset = get_dataset(dataset_name)

    if sequence is not None:
        dataset = [dataset[sequence]]

    trackers = [Tracker(tracker_name, tracker_param, dataset_name, run_id)]

    run_dataset(dataset, trackers, debug, threads, num_gpus=num_gpus)


def run_sequence_generation(seq: Sequence, tracker: Tracker, debug=False, num_gpu=8,noise=None,attack="rgb"):
    """Runs a tracker on a sequence."""
    '''2021.1.2 Add multiple gpu support'''
  
    try:
        worker_name = multiprocessing.current_process().name
        worker_id = int(worker_name[worker_name.find('-') + 1:]) - 1
        gpu_id = worker_id % num_gpu
        torch.cuda.set_device(gpu_id)
    except:
        pass

    def _results_exist():
        if seq.object_ids is None:
            if seq.dataset in ['trackingnet', 'got10k']:
                base_results_path = os.path.join(tracker.results_dir, seq.dataset, seq.name)
                bbox_file = '{}.txt'.format(base_results_path)
            else:
                bbox_file = '{}/{}.txt'.format(tracker.results_dir, seq.name)
            return os.path.isfile(bbox_file)
        else:
            bbox_files = ['{}/{}_{}.txt'.format(tracker.results_dir, seq.name, obj_id) for obj_id in seq.object_ids]
            missing = [not os.path.isfile(f) for f in bbox_files]
            return sum(missing) == 0

    if _results_exist() and not debug:
        print('FPS: {}'.format(-1))
        return

    print('Tracker: {} {} {} ,  Sequence: {}'.format(tracker.name, tracker.parameter_name, tracker.run_id, seq.name))
    if tracker.initialize_noise is not None:#确保每个epoch初始化的时候noise是来自上一个epoch
        noise = tracker.initialize_noise
    if debug:
        output,noise = tracker.run_sequence(seq, debug=debug,noise=noise,attack=attack)
    else:
        try:
            output,noise = tracker.run_sequence(seq, debug=debug,noise=noise,attack=attack)
        except Exception as e:
            print(e)
            return

    sys.stdout.flush()

    if isinstance(output['time'][0], (dict, OrderedDict)):
        exec_time = sum([sum(times.values()) for times in output['time']])
        num_frames = len(output['time'])
    else:
        exec_time = sum(output['time'])
        num_frames = len(output['time'])

    print('FPS: {}'.format(num_frames / exec_time))

    # if not debug:
    #     _save_tracker_output(seq, tracker, output)
    # 那我这里返回的noise岂不是空？
    return noise


def run_sequence_test(seq: Sequence, tracker: Tracker, debug=False, num_gpu=8,noise=None,attack="rgb"):
    """Runs a tracker on a sequence."""
    '''2021.1.2 Add multiple gpu support'''
  
    try:
        worker_name = multiprocessing.current_process().name
        worker_id = int(worker_name[worker_name.find('-') + 1:]) - 1
        gpu_id = worker_id % num_gpu
        torch.cuda.set_device(gpu_id)
    except:
        pass

    def _results_exist():
        if seq.object_ids is None:
            if seq.dataset in ['trackingnet', 'got10k']:
                base_results_path = os.path.join(tracker.results_dir, seq.dataset, seq.name)
                bbox_file = '{}.txt'.format(base_results_path)
            else:
                bbox_file = '{}/{}.txt'.format(tracker.results_dir, seq.name)
            return os.path.isfile(bbox_file)
        else:
            bbox_files = ['{}/{}_{}.txt'.format(tracker.results_dir, seq.name, obj_id) for obj_id in seq.object_ids]
            missing = [not os.path.isfile(f) for f in bbox_files]
            return sum(missing) == 0

    if _results_exist() and not debug:
        print('FPS: {}'.format(-1))
        return

    print('Tracker: {} {} {} ,  Sequence: {}'.format(tracker.name, tracker.parameter_name, tracker.run_id, seq.name))

    if debug:
        output_test,noise = tracker.run_sequence(seq, debug=debug,noise=noise,attack=attack)
    else:
        try:
            output_test,noise = tracker.run_sequence(seq, debug=debug,noise=noise,attack=attack)
        except Exception as e:
            print(e)
            return

    sys.stdout.flush()

    if isinstance(output_test['time'][0], (dict, OrderedDict)):
        exec_time = sum([sum(times.values()) for times in output_test['time']])
        num_frames = len(output_test['time'])
    else:
        exec_time = sum(output_test['time'])
        num_frames = len(output_test['time'])

    print('FPS: {}'.format(num_frames / exec_time))

    if not debug:
        _save_tracker_output(seq, tracker, output_test)

    return noise


def main():
    
    parser = argparse.ArgumentParser(description='Run tracker on sequence or dataset.')
    parser.add_argument('tracker_name', type=str, help='Name of tracking method.')
    parser.add_argument('tracker_param', type=str, help='Name of config file.')
    parser.add_argument('--runid', type=int, default=None, help='The run id.')
    parser.add_argument('--dataset_name', type=str, default='otb', help='Name of dataset.')
    parser.add_argument('--sequence', type=str, default=None, help='Sequence number or name.')
    parser.add_argument('--debug', type=int, default=0, help='Debug level.')
    parser.add_argument('--threads', type=int, default=0, help='Number of threads.')
    parser.add_argument('--num_gpus', type=int, default=8)

    args = parser.parse_args()

    try:
        seq_name = int(args.sequence)
    except:
        seq_name = args.sequence
    dataset = get_dataset(args.dataset_name)

    if seq_name is not None:
        dataset = [dataset[seq_name]]

    train_noise_trackers = [Tracker(args.tracker_name, args.tracker_param, args.dataset_name, args.runid)]

    test_noise_trackers = [Tracker(args.tracker_name, args.tracker_param, args.dataset_name, args.runid)]

    multiprocessing.set_start_method('spawn', force=True)

    print('Evaluating {:4d} trackers on {:5d} sequences'.format(len(train_noise_trackers), len(dataset)))
    dataset_start_time = time.time()

    multiprocessing.set_start_method('spawn', force=True)

    # param_list = [(seq, tracker_info, args.debug, args.num_gpus) for seq, tracker_info in product(dataset, train_noise_trackers)]
    attack = "rgb_guide"
    noise = None
    param_list_with_attack = [(seq, tracker_info, args.debug, args.num_gpus,noise,attack) for seq, tracker_info in product(dataset, train_noise_trackers)]

    for epoch in range(10):
        log_file = '/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c01/DATA/chenqiang/RGB_Event_Frame_Attack/event_CEUTrack/output/test/tracking_results/logs/attack_rgb.log'
        with open(log_file, 'a') as f:
                log_entry = f"epoch: {epoch}\n"
                f.write(log_entry)
        begin_time = time.time()
        with multiprocessing.Pool(processes=args.threads) as pool:
            noise = pool.starmap(run_sequence_generation, param_list_with_attack) # all test seq
            param_list_with_attack = [(seq, tracker_info, args.debug, args.num_gpus,noise[-1].detach().cpu().clone(),attack) for seq, tracker_info in product(dataset, train_noise_trackers)]
    
    print('Done, total time: {}'.format(str(timedelta(seconds=(time.time() - dataset_start_time)))))

    #读取配置文件 将其设置为oriinial
    attack = "attack_rgb"
    # 将生成的扰动进行攻击测试
    param_list_with_attack_test = [(seq, tracker_info, args.debug, args.num_gpus,noise[-1].detach().cpu().clone(),attack) for seq, tracker_info in product(dataset, test_noise_trackers)]

    begin_time = time.time()
    with multiprocessing.Pool(processes=args.threads) as pool:
        pool.starmap(run_sequence_test, param_list_with_attack_test) # all test seq
    print('Done, total time: {}'.format(str(timedelta(seconds=(time.time() - dataset_start_time)))))

if __name__ == '__main__':
    main()
