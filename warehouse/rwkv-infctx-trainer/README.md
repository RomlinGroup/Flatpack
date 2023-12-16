# rwkv-infctx-trainer

[![pip install flatpack](https://img.shields.io/badge/pip%20install-flatpack-5865f2)](https://pypi.org/project/flatpack/)

> :warning: **DISCLAIMER:** This repository contains our research. Verify the information and do your own research (DYOR). We assume no responsibility for accuracy or completeness.

`rwkv-infctx-trainer` offers a flatpack.ai adaptation of [RWKV-infctx-trainer](https://github.com/RWKV/RWKV-infctx-trainer/) by RWKV, licensed under the [Apache License 2.0](https://github.com/RWKV/RWKV-infctx-trainer/blob/main/LICENSE).

We have no official affiliation or association with RWKV, nor are we endorsed or authorised by them. For the official RWKV website, please visit [https://rwkv.com](https://rwkv.com).

flatpack.ai is experimental; please avoid using it for production.

## train.sh (scratch)

Dataset: [tinyshakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) ([Public Domain](https://creativecommons.org/publicdomain/zero/1.0/))

> 40,000 lines of Shakespeare from a variety of Shakespeare's plays. Featured in Andrej Karpathy's blog post 'The Unreasonable Effectiveness of Recurrent Neural Networks': http://karpathy.github.io/2015/05/21/rnn-effectiveness/ ([TensorFlow 2023](https://www.tensorflow.org/datasets/catalog/tiny_shakespeare)).
