<div align="center">
  <img src="https://raw.githubusercontent.com/romlingroup/flatpack-ai/main/client/static/images/flatpack_ai_logo.svg" width="100" height="100" alt="Flatpack AI">
</div>

# flatpack.ai

[![pip install flatpack](https://img.shields.io/badge/pip%20install-flatpack-5865f2)](https://pypi.org/project/flatpack/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/romlingroup/flatpack-ai/blob/main/notebooks/flatpack_ai_playground.ipynb)

> :warning: **DISCLAIMER:** This repository contains our research. Verify the information and do your own research (DYOR). We assume no responsibility for accuracy or completeness.

> 😱 **Regarding AI fearmongering:** ["At what specific date in the future, if the apocalypse hasn't happened, will you finally admit to being wrong?"](https://bigthink.com/pessimists-archive/ai-fear-overpopulation/) (Archie McKenzie 2023) / ["Fearmongering is a technique that has benefited many people over the ages."](https://www.youtube.com/watch?v=2ZbRKxZ2cjM) (Fred L. Smith, Jr. 2019)

(OPEN SOURCE) flatpack.ai will democratize AI by providing a modular and open platform for anyone to train their AI
models from scratch with cutting-edge technology accessible to all. What flatpacks did for the furniture industry, we
will do for the AI industry.

https://pypi.org/project/flatpack

```bash
pip install flatpack
```

## Flatpack (FPK) 📦

- template.fpk
  - DATASET.md
  - flatpack.toml
  - README.md
  - train.sh

### Code signing

Our platform uses [RSA keys](<https://en.wikipedia.org/wiki/RSA_(cryptosystem)>) to authenticate and safeguard the integrity of flatpacks (FPK) - packages that may encompass both code and data vital for training artificial intelligence models.

The unimpeachable integrity of these flatpacks is paramount, as any compromise could have far-reaching implications on the reliability and security of the AI models generated and deployed.

> **Security note:** This project utilizes 4096-bit RSA keys for code signing. Breaking such encryption with current classical computing resources is computationally prohibitive\*. The analogy often used is that it would be akin to cataloguing every star in the known universe - multiple times.

> \* ["Post-quantum cryptography"](https://en.wikipedia.org/wiki/Post-quantum_cryptography) (Wikipedia 2023) / ["Did China Break The Quantum Barrier?"](https://www.forbes.com/sites/arthurherman/2023/01/10/did-china-break-the-quantum-barrier/) (Arthur Herman 2023) / ["No, RSA Encryption Isn't Obsolete"](https://www.aei.org/foreign-and-defense-policy/no-rsa-encryption-isnt-obsolete/) (Jason Blessing 2023) / ["Toward Quantum Resilient Security Keys"](https://security.googleblog.com/2023/08/toward-quantum-resilient-security-keys.html) (Google 2023)

```bash
openssl genpkey -algorithm RSA -out private_key.pem -aes256 -pkeyopt rsa_keygen_bits:4096
```

```bash
openssl rsa -pubout -in private_key.pem -out public_key.pem
```

#### Command-line options

To run the test script, use the following command-line options:

- `-i` or `--input`: Specifies the input file or folder (e.g., `hello_world`).
- `-o` or `--output`: Specifies where the compressed file will be saved (e.g., `compressed_file.fpk`).
- `-s` or `--signed`: Specifies where the signed file will be saved (e.g., `signed_file.fpk`).
- `-p` or `--private_key`: Specifies the path to the private key used for code signing (e.g., `private_key.pem`).
- `--hash_size`: (Optional) Specifies the hash size for code signing. The default is `256`.
- `--passphrase`: (Optional) Specifies the passphrase for the private key. The default is `None`.

##### Example

```bash
PASSPHRASE=$(python -c 'import getpass; print(getpass.getpass("Enter the passphrase: "))')
python test_compress_and_sign_fpk.py -i hello_world -o compressed_file.fpk -s signed_file.fpk -p private_key.pem --hash_size 256 --passphrase "$PASSPHRASE"
```

##### Verification

```bash
python verify_signed_data_with_cli.py --signed_file signed_file.fpk --public_key public_key.pem
```

## DATASET.md 📖

[DATASET.md](https://github.com/romlingroup/flatpack-ai/blob/main/datasets/DATASET.md) is our meticulously crafted
Markdown template (under development), established to act as a standard for encapsulating the essentials of any dataset.
By streamlining the documentation process, this template ensures that all relevant details about the dataset are easily
accessible, facilitating seamless collaboration and utilization.

Markdown is ideal for documenting datasets as it is lightweight and easy to archive. Its format helps keep data and
documentation in sync, essential for research integrity and reproducibility. Markdown files and datasets can also be
version-controlled, ensuring a cohesive historical record.

Once completed, it will be a mandatory component in all flatpacks.

## Micro-LLMs 🤏

[Micro-LLMs](https://github.com/karpathy/llama2.c#contributing) (credit to Andrej Karpathy)\*, or scaled-down language
models with around 100 million to 10 billion parameters, offer a compelling solution to the deployment and scalability
challenges associated with
traditional [large language models (LLMs)](https://en.wikipedia.org/wiki/Large_language_model). Their smaller size
allows for rapid pre-training and fine-tuning, enabling organizations to adapt quickly to market changes or specific
requirements.

Micro-LLMs can be deployed in edge-adjacent environments like laptops, microprocessors, or smartphones, benefiting
from [edge computing](https://en.wikipedia.org/wiki/Edge_computing) (computation and storage closer to data sources) to
facilitate low-latency and privacy-conscious applications. These characteristics make micro-LLMs broadly accessible,
energy-efficient, and specialized, even to smaller teams or individual developers.

Overall, micro-LLMs represent a logical step in the evolution of language models, effectively merging the capabilities
of LLMs with the practical needs of real-world applications.

\* _We recognize that "micro-LLMs" are oxymoronic, combining 'micro' and 'large.' However, the term aptly captures the
essence of these scaled-down but still powerful versions of large language models._

## Inspiration (no affiliation) ⭐

> **Arduino:** "There was once a barrier between the electronics, design, and programming world and the rest of the world. Arduino has broken down that barrier." ([Arduino 2021](https://www.arduino.cc/en/about))

> **Colab:** "Anyone with an internet connection can access Colab, and use it free of charge. Millions of students use Colab every month to learn Python programming and machine learning." ([Google 2023](https://blog.google/technology/developers/google-colab-ai-coding-features/))

> **Micro-LLMs:** "Basically I think there will be a lot of interest in training or finetuning custom micro-LLMs (think ~100M - ~1B params, but let's say up to ~10B params) across a large diversity of applications, and deploying them in edge-adjacent environments (think MCUs, phones, web browsers, laptops, etc.)." ([Andrej Karpathy 2023](https://github.com/karpathy/llama2.c))

> **ONNX:** "Many people are working on great tools, but developers are often locked in to one framework or ecosystem. ONNX is the first step in enabling more of these tools to work together by allowing them to share models." ([ONNX 2023](https://onnx.ai/about.html))

> **Raspberry Pi Foundation:** "The Raspberry Pi Foundation is a UK-based charity with the mission to enable young people to realise their full potential through the power of computing and digital technologies." ([Raspberry Pi Foundation 2023](https://www.raspberrypi.org/about/))

> **Standardisation:** "Without standards, there can be no improvement." (Taiichi Ohno)

## Commercial use allowed (no affiliation) 📈

> ⚖️ **Legal perspectives:** ["Questions and Answers – New EU copyright rules"](https://ec.europa.eu/commission/presscorner/detail/en/qanda_21_2821) (European Commission 2021) / ["Are ChatGPT, Bard and Dolly 2.0 Trained On Pirated Content?"](https://www.searchenginejournal.com/are-chatgpt-bard-and-dolly-2-0-trained-on-pirated-content/) (Roger Montti 2023) / ["Llama copyright drama: Meta stops disclosing what data it uses to train the company's giant AI models"](https://www.businessinsider.com/meta-llama-2-data-train-ai-models-2023-7) (Alistair Barr 2023) / ["Generative Artificial Intelligence and Copyright Law"](https://crsreports.congress.gov/product/pdf/LSB/LSB10922) (Congressional Research Service 2023) / ["EU legislates disclosure of copyright data used to train AI"](https://www.theregister.com/2023/05/01/eu_ai_act_adds_new/) (Katyanna Quach 2023)

We constantly search for datasets and models suitable for future deployment
as [flatpacks](https://github.com/romlingroup/flatpack-ai/tree/main/warehouse) (FPK). Therefore, if you know of any
high-quality datasets or models with commercially viable licenses, we would appreciate it if you submitted them via a
pull request.

_Before utilizing any dataset or model for commercial purposes, seeking guidance from a legal adviser is crucial to
understand the legality within your jurisdiction. Unauthorized use of content may result in severe legal consequences.
Opt for datasets and models with transparent, commercially viable licenses, subject to review by legal experts.
Maintaining transparency about data sources is vital to address legal and ethical concerns. This list of provided models
or datasets is intended solely for research purposes; exercise due diligence by independently verifying their licensing
and authenticity before any commercial application. Consult legal counsel to ensure compliance with relevant laws and
regulations if needed._

| Name                                                                                    | Type    | License                                                                                                   |
|:----------------------------------------------------------------------------------------|:--------|:----------------------------------------------------------------------------------------------------------|
| [databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) | Dataset | [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/)                                           |
| [falcon-refinedweb](https://huggingface.co/datasets/tiiuae/falcon-refinedweb)           | Dataset | [ODC-By 1.0](https://opendatacommons.org/licenses/by/1-0/)                                                |
| [Cerebras-GPT-13B](https://huggingface.co/cerebras/Cerebras-GPT-13B)                    | Model   | [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)                                         |
| [falcon-7b](https://huggingface.co/tiiuae/falcon-7b)                                    | Model   | [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)                                         |
| [falcon-7b-instruct](https://huggingface.co/tiiuae/falcon-7b-instruct)                  | Model   | [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)                                         |
| [falcon-40b](https://huggingface.co/tiiuae/falcon-40b)                                  | Model   | [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)                                         |
| [falcon-40b-instruct](https://huggingface.co/tiiuae/falcon-40b-instruct)                | Model   | [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)                                         |
| [falcon-180B](https://huggingface.co/tiiuae/falcon-180B)                                | Model   | [Falcon-180B TII License](https://huggingface.co/spaces/tiiuae/falcon-180b-license/blob/main/LICENSE.txt) |
| [falcon-180B-chat](https://huggingface.co/tiiuae/falcon-180B-chat)                      | Model   | [Falcon-180B TII License](https://huggingface.co/spaces/tiiuae/falcon-180b-license/blob/main/LICENSE.txt) |
| [h2ogpt-oasst1-falcon-40b](https://huggingface.co/h2oai/h2ogpt-oasst1-falcon-40b)       | Model   | [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)                                         |
| [Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf)                        | Model   | [LLAMA 2 LICENSE](https://ai.meta.com/resources/models-and-libraries/llama-downloads/)                    |
| [Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)              | Model   | [LLAMA 2 LICENSE](https://ai.meta.com/resources/models-and-libraries/llama-downloads/)                    |
| [Llama-2-13b-hf](https://huggingface.co/meta-llama/Llama-2-13b-hf)                      | Model   | [LLAMA 2 LICENSE](https://ai.meta.com/resources/models-and-libraries/llama-downloads/)                    |
| [Llama-2-13b-chat-hf](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf)            | Model   | [LLAMA 2 LICENSE](https://ai.meta.com/resources/models-and-libraries/llama-downloads/)                    |
| [Llama-2-70b-hf](https://huggingface.co/meta-llama/Llama-2-70b-hf)                      | Model   | [LLAMA 2 LICENSE](https://ai.meta.com/resources/models-and-libraries/llama-downloads/)                    |
| [Llama-2-70b-chat-hf](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf)            | Model   | [LLAMA 2 LICENSE](https://ai.meta.com/resources/models-and-libraries/llama-downloads/)                    |
| [mpt-7b](https://huggingface.co/mosaicml/mpt-7b)                                        | Model   | [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)                                         |
| [mpt-7b-instruct](https://huggingface.co/mosaicml/mpt-7b-instruct)                      | Model   | [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/)                                           |
| [mpt-7b-storywriter](https://huggingface.co/mosaicml/mpt-7b-storywriter)                | Model   | [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)                                         |
| [open_llama_3b](https://huggingface.co/openlm-research/open_llama_3b)                   | Model   | [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)                                         |
| [open_llama_7b](https://huggingface.co/openlm-research/open_llama_7b)                   | Model   | [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)                                         |
| [open_llama_13b](https://huggingface.co/openlm-research/open_llama_13b)                 | Model   | [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)                                         |
| [pythia-12b](https://huggingface.co/EleutherAI/pythia-12b)                              | Model   | [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)                                         |
| [stablelm-base-alpha-3b](https://huggingface.co/stabilityai/stablelm-base-alpha-3b)     | Model   | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)                                           |
| [stablelm-base-alpha-7b](https://huggingface.co/stabilityai/stablelm-base-alpha-7b)     | Model   | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)                                           |

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Friendly notice ❤️

The flatpack.ai logo belongs to [Romlin Group AB](https://romlin.com) and is protected by copyright. Please note that
the logo is NOT covered by the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0) that applies to the
source code in this repository. If you want to use the logo for anything besides GitHub shenanigans, please ask for
permission first. Thanks a million for being super awesome and respectful!