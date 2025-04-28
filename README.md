# ArtQuantization

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

### AWQ

It takes me about 1 hour to quantize Qwen2.5-32B-Instruct by AWQ algorithm.

The performance is good, however, Tesla-V100 do not support AWQ model.