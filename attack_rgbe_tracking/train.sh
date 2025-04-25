export CUDA_VISIBLE_DEVICES=3,4
python /media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c01/DATA/chenqiang/RGB_Event_Frame_Attack/event_CEUTrack/tracking/train.py --script ceutrack --config ceutrack_coesot  \
 --save_dir ./output --mode multiple --nproc_per_node 2 --use_wandb  0