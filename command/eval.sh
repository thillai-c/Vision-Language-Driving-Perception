python3 -m torch.distributed.run \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=1 \
    --master_port=1246 \
    internvl_eval.py \
    --checkpoint ../data/bpv_finetune/checkpoint-550 \
    --data_file ../data/dataset/vlm_ann_eval.jsonl \
    --out-dir vlm_eval \
    --temperature 0.3 \
    --mode slow \
    --root ../data/nuplan/dataset/nuplan-v1.1/sensor_blobs