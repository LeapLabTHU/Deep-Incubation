# Training
We provide the instructions for training ViT-B, ViT-L and ViT-H on ImageNet-1K here.

- You can add `--use_amp 1` to train in PyTorch's Automatic Mixed Precision (AMP).
- Auto-resuming is enabled by default, i.e., the training script will automatically resume from the latest ckpt in <code>output_dir</code>.
- The effective batch size = `NGPUS` * `batch_size` * `update_freq`. We keep using an effective batch size of 2048. To avoid OOM issues, you may adjust these arguments accordingly.
- We provide single-node training scripts for simplicity. For multi-node training, simply modify the training scripts accordingly with torchrun:

```bash
python -m torch.distributed.launch --nproc_per_node=${NGPUS} --master_port=23346 --use_env main.py ...

# modify the above code to

torchrun \
--nnodes=$NODES \
--nproc_per_node=$NGPUS \
--rdzv_backend=c10d \
--rdzv_endpoint=$MASTER_ADDR:60900 \
main.py ...
```


## Preparing meta models

We have released our pre-trained meta models at [🤗 Hugging Face](https://huggingface.co/nzl-thu/Model-Assembling/tree/main/log_dir), which can be directly used for modular training.

|   model    |                                                    checkpoint                                                     |
|:----------:|:-----------------------------------------------------------------------------------------------------------------:|
| ViT-B-Meta |  [🤗 HF link](https://huggingface.co/nzl-thu/Model-Assembling/blob/main/log_dir/PT_base/finished_checkpoint.pth)  |
| ViT-L-Meta | [🤗 HF link]( https://huggingface.co/nzl-thu/Model-Assembling/blob/main/log_dir/PT_large/finished_checkpoint.pth) |
| ViT-H-Meta | [🤗 HF link]( https://huggingface.co/nzl-thu/Model-Assembling/blob/main/log_dir/PT_huge/finished_checkpoint.pth)  |

To train ViT-B for example, simply download the ViT-B-Meta model and save it to <code>./log_dir/PT_base/finished_checkpoint.pth</code>. Then follow the training script below for modular training.

<details>
<summary>Alternatively, you can also pre-train the meta models from scratch (click to expand)</summary>

```bash
PHASE=PT
MODEL=base  # for base
# MODEL=large  # for large
# MODEL=huge  # for huge
NGPUS=8

args=(
--phase ${PHASE} 
--model vit_${MODEL}_patch16_224   # for base, large
# --model vit_${MODEL}_patch14_224   # for huge
--divided_depths 1 1 1 1 
--output_dir ./log_dir/${PHASE}_${MODEL}

--batch_size 256
--epochs 300 
--drop-path 0 
)

python -m torch.distributed.launch --nproc_per_node=${NGPUS} --master_port=23346 --use_env main.py "${args[@]}"
```

</details>


## Modular training
Here we provide the modular training script. Note that the training process of each module can be executed **in parallel**.

```bash
PHASE=MT
MODEL=base DEPTH=12  # for base
# MODEL=large DEPTH=24  # for large
# MODEL=huge DEPTH=32  # for huge
NGPUS=8

args=(
--phase ${PHASE} 
--model vit_${MODEL}_patch16_224   # for base, large
# --model vit_${MODEL}_patch14_224   # for huge
--meta_model ./log_dir/PT_${MODEL}/finished_checkpoint.pth  # loading the pre-trained meta model

--batch_size 128
--update_freq 2
--epochs 100 
--drop-path 0.1
)

# Modular training each target module. Each line can be executed in parallel.
python -m torch.distributed.launch --nproc_per_node=${NGPUS} --master_port=23346 --use_env main.py "${args[@]}" --idx 0 --divided_depths $((DEPTH/4)) 1 1 1 --output_dir ./log_dir/${PHASE}_${MODEL}_0
python -m torch.distributed.launch --nproc_per_node=${NGPUS} --master_port=23346 --use_env main.py "${args[@]}" --idx 1 --divided_depths 1 $((DEPTH/4)) 1 1 --output_dir ./log_dir/${PHASE}_${MODEL}_1
python -m torch.distributed.launch --nproc_per_node=${NGPUS} --master_port=23346 --use_env main.py "${args[@]}" --idx 2 --divided_depths 1 1 $((DEPTH/4)) 1 --output_dir ./log_dir/${PHASE}_${MODEL}_2
python -m torch.distributed.launch --nproc_per_node=${NGPUS} --master_port=23346 --use_env main.py "${args[@]}" --idx 3 --divided_depths 1 1 1 $((DEPTH/4)) --output_dir ./log_dir/${PHASE}_${MODEL}_3

```

## Assemble & Fine-tuning

```bash
PHASE=FT
MODEL=base DEPTH=12  # for base
# MODEL=large DEPTH=24  # for large
# MODEL=huge DEPTH=32  # for huge
NGPUS=8

args=(
--phase ${PHASE} 
--model vit_${MODEL}_patch16_224   # for base, large
# --model vit_${MODEL}_patch14_224   # for huge
--incubation_models ./log_dir/MT_${MODEL}_*/finished_checkpoint.pth  # for assembling
--divided_depths $((DEPTH/4)) $((DEPTH/4)) $((DEPTH/4)) $((DEPTH/4)) \
--output_dir ./log_dir/${PHASE}_${MODEL}

--batch_size 64
--update_freq 4
--epochs 100 
--warmup-epochs 0
--clip-grad 1
--drop-path 0.1  # for base
# --drop-path 0.5  # for large
# --drop-path 0.6  # for huge
)

python -m torch.distributed.launch --nproc_per_node=${NGPUS} --master_port=23346 --use_env main.py "${args[@]}"
```
