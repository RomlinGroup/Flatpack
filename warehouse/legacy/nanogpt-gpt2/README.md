# nanogpt-gpt2
[![pip install flatpack](https://img.shields.io/badge/pip%20install-flatpack-5865f2)](https://pypi.org/project/flatpack/)

> :warning: **DISCLAIMER:** This repository contains our research. Verify the information and do your own research (DYOR). We assume no responsibility for accuracy or completeness.

`nanogpt-gpt2` offers a flatpack.ai adaptation of [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy, licensed under the [MIT License](https://github.com/karpathy/nanoGPT/blob/master/LICENSE).

We have no official affiliation or association with Andrej Karpathy, nor are we endorsed or authorized by him. For the official Andrej Karpathy website, please visit [https://karpathy.ai](https://karpathy.ai).

flatpack.ai is experimental; please avoid using it for production.

## train.sh (fine-tuning)

Model: [gpt2-xl](https://huggingface.co/gpt2-xl) ([Modified MIT License](https://github.com/openai/gpt-2/blob/master/LICENSE))
> GPT-2 XL is the 1.5B parameter version of GPT-2, a transformer-based language model created and released by OpenAI. The model is a pretrained model on English language using a causal language modeling (CLM) objective ([GPT-2 XL 2023](https://huggingface.co/gpt2-xl)).

Dataset: [tinyshakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) ([Public Domain](https://creativecommons.org/publicdomain/zero/1.0/))
> 40,000 lines of Shakespeare from a variety of Shakespeare's plays. Featured in Andrej Karpathy's blog post 'The Unreasonable Effectiveness of Recurrent Neural Networks': http://karpathy.github.io/2015/05/21/rnn-effectiveness/ ([TensorFlow 2023](https://www.tensorflow.org/datasets/catalog/tiny_shakespeare)).