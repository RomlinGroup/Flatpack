# gpt2-knowledge-distillation
[![pip install flatpack](https://img.shields.io/badge/pip%20install-flatpack-5865f2)](https://pypi.org/project/flatpack/)

> :warning: **DISCLAIMER:** This repository contains our research. Verify the information and do your own research (DYOR). We assume no responsibility for accuracy or completeness.

`gpt2-knowledge-distillation` offers a flatpack.ai adaptation of [GPT2-Knowledge-Distillation](https://github.com/ThuanNaN/GPT2-Knowledge-Distillation) by Thuan Nguyen Duong, licensed under the [MIT License](https://github.com/ThuanNaN/GPT2-Knowledge-Distillation/blob/main/LICENSE).

We have no official affiliation or association with Thuan Nguyen Duong, nor are we endorsed or authorized by him. For the official GitHub profile of Thuan Nguyen Duong, please visit [https://github.com/ThuanNaN](https://github.com/ThuanNaN).

flatpack.ai is experimental; please avoid using it for production.

## train.sh

Model: [gpt2](https://huggingface.co/gpt2) ([Modified MIT License](https://github.com/openai/gpt-2/blob/master/LICENSE))
> This model was developed by researchers at OpenAI to help us understand how the capabilities of language model capabilities scale as a function of the size of the models (by parameter count) combined with very large internet-scale datasets (WebText) ([OpenAI 2019](https://github.com/openai/gpt-2/blob/master/model_card.md)).

Dataset: [tinyshakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) ([Public Domain](https://creativecommons.org/publicdomain/zero/1.0/))
> 40,000 lines of Shakespeare from a variety of Shakespeare's plays. Featured in Andrej Karpathy's blog post 'The Unreasonable Effectiveness of Recurrent Neural Networks': http://karpathy.github.io/2015/05/21/rnn-effectiveness/ ([TensorFlow 2023](https://www.tensorflow.org/datasets/catalog/tiny_shakespeare)).