# InternVL2-1B Finetune docs

## Finetune（commmand/InternVL2-1B.sh）

### Download InternVL2-1B

Create a directory `ckpts/InternVL2-1B` under `Vision Language Driving Perception`
```bash
mkdir ckpts/InternVL2-1B
cd ckpts/InternVL2-1B
```

Install `huggingface_hub` to download `InternVL2-1B` pre-trained model
```bash
pip install -U huggingface_hub==0.25.0
export HF_ENDPOINT=https://hf-mirror.com
```

Log in and download the `InternVL2-1B` pre-trained model
```bash
# login
huggingface-cli login
# Download
huggingface-cli download --resume-download OpenGVLab/InternVL2-1B --local-dir ./
```

### Setting parameters

In command/InternVL2-1B, specify the path of the pre-trained model of `InternVL2-1B`, the path to save the model after fine-tuning, and the path to configure the fine-tuning data.
```bash
--model_name_or_path "ckpts/InternVL2-1B"
OUTPUT_DIR='data/bpv_finetune'
--meta_path "data/meta_config.json"
```

Configure the `root` path and `annotation` path in `data/meta_config.json`
```json
{
"bpv_data": {
	"root": "../data/nuplan/dataset/nuplan-v1.1/sensor_blobs",
	"annotation": "../data/dataset/vlm_ann_train.jsonl",
	"data_augment": false,
	"repeat_time": 1,
	"length": 10
	}
}
```

**Parameter Description**:
- `root`: Path to save sensor data in nuplan dataset
- `annotation`: Path to annotation file for fine-tuning VLM model
- `data_augment`: Used to control whether to perform data augmentation operation (default false)
- `repeat_time`: Number of times data is reused
- `length`: Amount of data in dataset

### FineTune

Run the fine-tuning program. The model after fine-tuning is saved in `../data/bpv_finetune`
```bash
GPUS=8 PER_DEVICE_BATCH_SIZE=128 sh command/InternVL2-1B.sh
```

## Evaluate(command/eval.sh)

### Setting parameters
In `command/eval.sh`, set the relevant parameters

```python
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
```

**Parameter Description**
- `--nnodes`: specifies the number of nodes participating in the distributed evaluation
- `--node_rank`: determines the ranking of the current node among all nodes participating in the distributed operation
- `--master_addr`: defines the address of the master node
- `--nproc_per_node`: sets the number of processes used on each node
- `--master_port`: specifies the port number used by the master node for communication
- `internvl_eval.py`: evaluation script path
- `--checkpoint`: path to save the fine-tuned model
- `--data_file`: path to the dataset used for evaluation
- `--out-dir`: path to save the evaluation results
- `--temperature`: used to control the randomness of the generated results
- `--mode`: when the mode is slow, the maximum number of output tokens is 100, and when it is fast, the maximum number of output tokens is 10
- `--root`: path to save sensor data in the nuplan dataset

### Usage
```bash
sh command/eval.sh
```


## PS：Parameter description in command/InternVL2-1B.sh

- `GPUS=${GPUS:-8}`: Number of GPUs used, default is 8
- `BATCH_SIZE=${BATCH_SIZE:-128}`: Batch size of training data, default is 128
- `PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-4}`: Batch size processed on each GPU device, default is 4
- `GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))`: Calculate the number of steps for gradient accumulation
- `export NCCL_P2P_DISABLE=1`: Disable P2P communication for 40 series graphics cards, comment out this sentence for other series graphics cards
- `export NCCL_IB_DISABLE=1`: Disable IB communication for 40 series graphics cards, comment out this sentence for other series graphics cards
- `export PYTHONPATH="${PYTHONPATH}:$(pwd)"`: Set the `python` path
- `export MASTER_PORT=34229`: Set the main port number in the distributed training environment to `34229`
- `export LAUNCHER=pytorch`: Specifies the launcher as `pytorch`
- `OUTPUT_DIR='../data/bpv_finetune'`: Used to specify the output directory for training results and other data
- `if [ ! -d "$OUTPUT_DIR" ]; then mkdir -p "$OUTPUT_DIR" fi`: Conditional statement to check whether the directory specified by `OUTPUT_DIR` exists
- `--nnodes`: The number of nodes involved in distributed training
- `--node_rank`: Determine the ranking of the current node among all nodes participating in distributed training
- `--master_addr`: Define the address of the master node
- `--nproc_per_node`: Set the number of GPUs used on each node
- `--master_port`: Determine the port number used by the master node
- `internvl/train/internvl_chat_finetune.py`: Path to the training script
- `--model_name_or_path`: Specify the path to the pre-trained model
- `--conv_style`: Parameters related to the model's conversation style
- `--output_dir`: Used to specify the output directory for data such as training results
- `--meta_path`: Specify the configuration file path for training data
- `--overwrite_output_dir`: Whether to overwrite files in `output_dir` (default is True)
- `--force_image_size`: Set the input size of the image
- `--max_dynamic_patch`: Set the maximum dynamic patch
- `--down_sample_ratio`: Set the downsampling ratio
- `--drop_path_rate`: Set the ratio of `drop_path`, `drop_path` is a regularization method
- `--freeze_llm`: Set whether to freeze the parameters of `LLM`
- `--freeze_mlp`: Set whether to freeze the parameters of `mlp`
- `--freeze_backbone`: Set whether to freeze the parameters of `backbone`
- `--vision_select_layer`: Set the layer of the selected vision model
- `--dataloader_num_workers`: Set the number of threads of the data loader
- `--bf16`: Set whether to use half-precision floating point numbers for calculation
- `--num_train_epochs`: Set the number of training rounds
- `--per_device_train_batch_size`: Set the amount of data processed by each GPU during each training
- `--evaluation_strategy`: Set the evaluation strategy
- `--save_strategy`: Set the strategy for saving the model
- `--save_steps`: Set the frequency of saving the model
- `--save_total_limit`: Set the total limit for saving
- `--learning_rate`: Set learning rate
- `--weight_decay`: Set weight decay rate
- `--warmup_ratio`: Set warmup ratio
- `--lr_scheduler_type`: Set learning rate scheduler
- `--logging_steps`: Set logging step interval
- `--max_seq_length`: Set maximum sequence length of reply
- `--do_train`: Explicitly perform training operation
- `--grad_checkpoint`: Set whether to use gradient checkpoint technology to reduce memory requirements
- `--group_by_length`: Used to group data by length
- `--dynamic_image_size`: Set whether image size changes dynamically
- `--use_thumbnail`: Set whether to use thumbnails
- `--ps_version`: Set the version of `pixel shuffle`
- `--deepspeed`: Set the configuration file for `Deepspeed` framework optimization training
- `--report_to`: Set `TensorBoard` to monitor the training process