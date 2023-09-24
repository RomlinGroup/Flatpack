# gpt2-knowledge-distillation
[![pip install flatpack](https://img.shields.io/badge/pip%20install-flatpack-5865f2)](https://pypi.org/project/flatpack/)

> :warning: **DISCLAIMER:** This repository contains our research. Verify the information and do your own research (DYOR). We assume no responsibility for accuracy or completeness.

`llama2-scratch` offers a flatpack.ai adaptation of [llama2.c](https://github.com/karpathy/llama2.c) by Andrej Karpathy, licensed under the [MIT License](https://github.com/karpathy/llama2.c/blob/master/LICENSE).

We have no official affiliation or association with Andrej Karpathy, nor are we endorsed or authorized by him. For the official Andrej Karpathy website, please visit [https://karpathy.ai](https://karpathy.ai).

flatpack.ai is experimental; please avoid using it for production.

## train.sh (scratch)

Dataset: [roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) ([CDLA-Sharing-1.0](https://cdla.dev/sharing-1-0/))
> Dataset containing synthetically generated (by GPT-3.5 and GPT-4) short stories that only use a small vocabulary. Described in the following paper: https://arxiv.org/abs/2305.07759 ([Ronen Eldan 2023](https://huggingface.co/datasets/roneneldan/TinyStories)).