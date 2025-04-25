export CUDA_VISIBLE_DEVICES=7
export PYTHONPATH=/AttackTracking/CEUTrack:$PYTHONPATH
python /AttackTracking/CEUTrack/tracking/test.py \
ceutrack ceutrack_coesot --dataset coesot --threads 1 --num_gpus 1
