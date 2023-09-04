# llama-efficient-tuning
[![pip install flatpack](https://img.shields.io/badge/pip%20install-flatpack-5865f2)](https://pypi.org/project/flatpack/)

> :warning: **DISCLAIMER:** This repository contains our research. Verify the information and do your own research (
> DYOR). We assume no responsibility for accuracy or completeness.

`llama-efficient-tuning` offers a flatpack.ai adaptation of [LLaMA-Efficient-Tuning](https://github.com/hiyouga/LLaMA-Efficient-Tuning) by Yaowei Zheng, licensed under the [Apache License 2.0](https://github.com/hiyouga/LLaMA-Efficient-Tuning/blob/main/LICENSE).

We have no official affiliation or association with Yaowei Zheng, nor are we endorsed or authorized by him. For the official Google Scholar profile of Yaowei Zheng, please visit [https://scholar.google.com/citations?user=QQtacXUAAAAJ&hl=en](https://scholar.google.com/citations?user=QQtacXUAAAAJ&hl=en).

flatpack.ai is experimental; please avoid using it for production.

## train.sh (fine-tuning)

Model: [tiiuae/falcon-7b](https://huggingface.co/tiiuae/falcon-7b) ([Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0))
> Falcon-7B is a 7B parameters causal decoder-only model built by [TII](https://www.tii.ae/) and trained on 1,500B tokens of [RefinedWeb](https://huggingface.co/datasets/tiiuae/falcon-refinedweb) enhanced with curated corpora. It is made available under the Apache 2.0 license ([Technology Innovation Institute 2023](https://huggingface.co/tiiuae/falcon-7b)).

Dataset: [wiki_demo](https://github.com/hiyouga/LLaMA-Efficient-Tuning/blob/main/data/wiki_demo.txt) ([CC BY-SA 4.0](https://en.wikipedia.org/wiki/Wikipedia:Text_of_the_Creative_Commons_Attribution-ShareAlike_4.0_International_License))