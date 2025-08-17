# VLM Converter

## Env Install 

```bash
conda activate vision-language-driving-perception
sudo apt-get -y install libopenmpi-dev && pip3 install --upgrade pip setuptools && pip3 install tensorrt_llm --extra-index-url https://pypi.nvidia.com
```

Configure domestic mirror source:

```bash
pip config set global.index-url https://mirrors.bfsu.edu.cn/pypi/web/simple
```

Installation dependencies:

```bash
cd /workspace
pip install -v -e .
```

## Pytorch Inference

```bash
GPUS=1 python3 -m torch.distributed.run \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=1 \
    --master_port=1246 \
    /workspace/pytorch_internvl_infer.py \
    --checkpoint /workspace/ckpts \
    --data_file /workspace/sample.jsonl \
    --out-dir vlm_eval \
    --temperature 0.3 \
    --mode slow \
    --root /workspace/vlm_samples
```

## VLM -> TensorRT

### ckpt converter

```python
python3 /workspace/vlmtrt/convert_qwen2_ckpt.py --model_dir /workspace/ckpts \
--output_dir /workspace/ckpts/tllm_checkpoint/ --dtype bfloat16
```

### engine builder

```python
trtllm-build --checkpoint_dir /workspace/ckpts/tllm_checkpoint/ \
--output_dir /workspace/ckpts/trt_engines/ \
--gemm_plugin bfloat16 --max_batch_size 2 --paged_kv_cache enable \
--max_input_len 32768 --max_seq_len 36960 --max_num_tokens 32768 --max_multimodal_len 13312
```

### ViT engine builder

```python
python3 /workspace/vlmtrt/build_vit_engine.py --pretrainedModelPath /workspace/ckpts/ \
--imagePath /workspace/frame_0.jpg \
--onnxFile /workspace/ckpts/vision_encoder_bfp16.onnx \
--trtFile /workspace/ckpts/vision_encoder_bfp16.trt \
--dtype bfloat16 --minBS 1 --optBS 13 --maxBS 26
```

## TensorRT Inference

```bash
python3 /workspace/trt_internvl_infer.py \
    --max_new_tokens 100 \
    --vit_engine_path /workspace/ckpts/trt_engines/vision_encoder_bfp16.trt \
    --intern_engine_dir /workspace/ckpts/trt_engines \
    --tokenizer_dir /workspace/ckpts \
    --data_file /workspace/sample.jsonl \
    --root /workspace/vlm_samples \
    --batch_size 1 \
    --num_workers 4 \
    --num_beams 1 \
    --max_new_tokens 10
```