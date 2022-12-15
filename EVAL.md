# Pre-trained Models & Evaluation
We have released our pre-trained models at [🤗 Hugging Face](https://huggingface.co/nzl-thu/Model-Assembling/tree/main/pretrained). The detailed information of each model is listed below:

| model | image size | #param. | top-1 acc. | checkpoint                                                                                        |
|-------|------------|---------|------------|---------------------------------------------------------------------------------------------------|
| ViT-B | 224x224        | 87M     | 82.4%      | [🤗 HF   link](https://huggingface.co/nzl-thu/Model-Assembling/blob/main/pretrained/vit_base.pth)  |
| ViT-L | 224x224        | 304M    | 83.9%      | [🤗 HF   link](https://huggingface.co/nzl-thu/Model-Assembling/blob/main/pretrained/vit_large.pth) |
| ViT-H | 224x224        | 632M    | 84.3%      | [🤗 HF   link](https://huggingface.co/nzl-thu/Model-Assembling/blob/main/pretrained/vit_huge.pth)  |

To evaluate ViT-B for example, simply download the pre-trained ViT-B model and save it to <code>./pretrained/vit_base.pth</code>. Then use the script below for evaluation.

```bash
MODEL=base DEPTH=12  # for base
# MODEL=large DEPTH=24  # for large
# MODEL=huge DEPTH=32  # for huge

args=(
--model vit_${MODEL}_patch16_224   # for base, large
# --model vit_${MODEL}_patch14_224   # for huge
--finetune ./pretrained/vit_${MODEL}.pth
--divided_depths $((DEPTH/4)) $((DEPTH/4)) $((DEPTH/4)) $((DEPTH/4)) \

--eval
--batch_size 1024
)

python -m torch.distributed.launch --nproc_per_node=1 --master_port=23346 --use_env main.py "${args[@]}"
```

Expected results:

```bash
# base
* Acc@1 82.368 Acc@5 96.050 loss 0.782
# large
* Acc@1 83.894 Acc@5 96.702 loss 0.705
# huge
* Acc@1 84.310 Acc@5 96.810 loss 0.709
```
