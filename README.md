# ArtQuantization

## Requirements

```sh
conda create -n quantization python=3.12 -y
conda activate quantization
pip install peft
pip install autoawq
pip install llmcompressor
```

## Evaluation Results

Qwen2.5-7B-Instruct:

* 2 * RTX3090: 8G for the maximum one
* 4 * RTX3090: 4G for the maximum one
* 8 * RTX3090: 3G for the maximum one

Qwen2.5-32B-Instruct:

* 4 * RTX3090: 17G for the maximum one
* 8 * RTX3090: 8G for the maximum one

Qwen2.5-32B-Instruct-AWQ:

* 2 * RTX3090: 11G for the maximum one
* 4 * RTX3090: 6G for the maximum one
* 8 * RTX3090: 4G for the maximum one

Qwen2.5-32B-Instruct-GPTQ-INT8:

* 2 * RTX3090: OOM
* 4 * RTX3090: 25G for the maximum one
* 8 * RTX3090: 15G for the maximum one

Qwen2.5-32B-Instruct-GPTQ-W4A16:

* 2 * RTX3090: OOM
* 4 * RTX3090: 25G for the maximum one
* 8 * RTX3090: 15G for the maximum one

Qwen2.5-32B-Instruct-GPTQ-W4A8:

* 2 * RTX3090: OOM
* 4 * RTX3090: 24G for the maximum one
* 8 * RTX3090: 14G for the maximum one

### AWQ

It takes me about 1 hour to quantize Qwen2.5-32B-Instruct by AWQ algorithm.

The performance is good, however, Tesla-V100 do not support AWQ model. 

### GPTQ

It takes me about 4 hour to quantize Qwen2.5-32B-Instruct by GPTQ-INT4 algorithm.

But the memory usage for quantized model is even larger than that for base model.
