# lit-gpt-falcon
[![pip install flatpack](https://img.shields.io/badge/pip%20install-flatpack-5865f2)](https://pypi.org/project/flatpack/)

> :warning: **DISCLAIMER:** This repository contains our research. Verify the information and do your own research (DYOR). We assume no responsibility for accuracy or completeness.

`lit-gpt-falcon` offers a flatpack.ai adaptation of [Lit-GPT](https://github.com/Lightning-AI/lit-gpt) by Lightning AI, licensed under the [Apache License 2.0](https://github.com/Lightning-AI/lit-gpt/blob/main/LICENSE).

We have no official affiliation or association with Lightning AI, nor are we endorsed or authorized by them or any of their subsidiaries or affiliates. For the official Lightning AI website, please visit https://lightning.ai.

flatpack.ai is experimental; please avoid using it for production.

## train.sh (fine-tuning)

Model: [tiiuae/falcon-7b](https://huggingface.co/tiiuae/falcon-7b) ([Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0))
> Falcon-7B is a 7B parameters causal decoder-only model built by [TII](https://www.tii.ae/) and trained on 1,500B tokens of [RefinedWeb](https://huggingface.co/datasets/tiiuae/falcon-refinedweb) enhanced with curated corpora. It is made available under the Apache 2.0 license ([Technology Innovation Institute 2023](https://huggingface.co/tiiuae/falcon-7b)).

Dataset: [alpaca_data_cleaned_archive.json](https://raw.githubusercontent.com/tloen/alpaca-lora/main/alpaca_data_cleaned_archive.json) ([CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/))
> Alpaca is intended and licensed for research use only. The dataset is CC BY NC 4.0 (allowing only non-commercial use) and models trained using the dataset should not be used outside of research purposes. The weight diff is also CC BY NC 4.0 (allowing only non-commercial use) ([Stanford Alpaca 2023](https://github.com/tatsu-lab/stanford_alpaca)).