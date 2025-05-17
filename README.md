# ArtQuantization

This repository contains the implementation of ArtQuantization, a novel approach for model quantization that combines GPTQ with Shapley value correction.

## Features

- Multiple quantization methods:
  - GPTQ-based quantization
  - OBS (Optimal Brain Surgeon) quantization
  - Shapley value-based quantization
- Support for various model architectures
- Configurable quantization parameters

## Requirements

```sh
conda create -n quantization python=3.12 -y
conda activate quantization
pip install peft
pip install autoawq
pip install llmcompressor
```



## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.