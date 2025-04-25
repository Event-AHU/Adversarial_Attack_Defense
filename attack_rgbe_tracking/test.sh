export CUDA_VISIBLE_DEVICES=6
export PYTHONPATH=/RGB_Event_Frame_Attack/event_CEUTrack:$PYTHONPATH
python /RGB_Event_Frame_Attack/event_CEUTrack/tracking/test.py \
ceutrack ceutrack_coesot --dataset coesot --threads 1 --num_gpus 1