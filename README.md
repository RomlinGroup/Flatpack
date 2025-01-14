<p align="center">
  <img src="https://romlin.com/wp-content/uploads/2023/05/flatpack_ai_logo.svg" width="100" height="100" alt="Flatpack">
</p>

# Flatpack

[![PyPI - Version](https://img.shields.io/pypi/v/flatpack)](https://pypi.org/project/flatpack/) [![PyPI - Downloads](https://img.shields.io/pypi/dm/flatpack)](https://pypistats.org/packages/flatpack) [![DeepSource](https://app.deepsource.com/gh/RomlinGroup/Flatpack.svg/?label=active+issues&show_trend=false&token=ludXWV8MdvRbiETqDIORwomF)](https://app.deepsource.com/gh/RomlinGroup/Flatpack/)

> **DISCLAIMER:** This repository contains our research. Verify the information and do your own research (DYOR). We assume no responsibility for accuracy or completeness.

> **免责声明:** 此存储库包含我们的研究。请核实信息并自行研究 (DYOR)。我们不对准确性或完整性承担任何责任。

> **Regarding AI fearmongering:** ["At what specific date in the future, if the apocalypse hasn't happened, will you finally admit to being wrong?"](https://bigthink.com/pessimists-archive/ai-fear-overpopulation/) (Archie McKenzie 2023) / ["Fearmongering is a technique that has benefited many people over the ages."](https://www.youtube.com/watch?v=2ZbRKxZ2cjM) (Fred L. Smith, Jr. 2019)

## Ready-to-assemble AI

<p align="center">
  <img src="https://fpk.ai/assets/terminal.svg" width="600">
</p>

At Flatpack, our mission is clear: we are committed to building trust in AI.

Flatpack democratises AI and ML through micro language models and model compression. Our platform enables users to train custom language models with 100M to 10B parameters. We introduce flatpacks (FPKs) to integrate AI and ML into edge computing, electronic components, and robots.

在 Flatpack，我们的使命十分明确: 致力于建立对 AI 的信任。

Flatpack 通过微语言模型和模型压缩普及 AI 和 ML。我们的平台使用户能够训练具有 100M 到 10B 个参数的自定义语言模型。我们引入了 flatpacks (FPKs)，将 AI 和 ML 集成到边缘计算、电子元件和机器人中。

> **Flatpack:** "Picture merging the Swedish ingenuity of flatpacks (hence our name) and ready-to-assemble furniture with the imaginative appeal of certain Danish snap-together toy bricks. Our method of developing micro language models is designed to be intelligently integrated into the physical world."

## Disclaimer

Flatpack is a general-purpose generative AI platform that supports educational, experimental, and low-risk applications, offering flexible tools for innovation and research. Flatpack is not intended for deployment in high-stakes or regulated environments, such as critical infrastructure, healthcare, law enforcement, or other areas classified as high-risk under the [EU AI Act](https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai) (Regulation (EU) 2024/1689).

Users are solely responsible for conducting thorough compliance assessments and ensuring that any use of Flatpack adheres to applicable laws, including the EU AI Act and related regulations.

## Edge artificial intelligence

Edge artificial intelligence uses local devices to enhance decision-making near data sources, improving privacy, response times,  and security while reducing reliance on cloud connectivity. Benefits include decreased latency, improved scalability, and reduced energy usage.

边缘人工智能使用本地设备来增强数据源附近的决策能力，提高隐私性、响应时间和安全性，同时减少对云连接的依赖，包括降低延迟、提高可扩展性和减少能源消耗等优势。

> "Edge artificial intelligence (AI), or AI at the edge, is the implementation of artificial intelligence in an edge computing environment, which allows computations to be done close to where data is actually collected, rather than at a centralized cloud computing facility or an offsite data center." ([Red Hat 2023](https://www.redhat.com/en/topics/edge-computing/what-is-edge-ai))

## Micro language models

[Micro language models](http://microlanguagemodels.com) (100M to 10B parameters) provide an efficient alternative to large language models, addressing deployment, reliability, and scalability challenges. Their compact size enables rapid pre-training and fine-tuning, allowing organisations to adapt swiftly to market shifts and specific needs. These models can be readily deployed in edge-adjacent environments such as laptops, microprocessors, and smartphones, offering accessibility and versatility.

[微型语言模型](http://microlanguagemodels.com)（100M 到 10B 个参数）为大型语言模型提供了一种有效的替代方案，解决了部署、可靠性和可扩展性挑战。其紧凑尺寸可实现快速预训练和微调，使组织能够迅速适应市场变化和特定需求。这些模型可轻松部署在笔记本电脑、微处理器和智能手机等边缘相邻环境中，提供可访问性及多功能性。

## Flatpack 3.X.X (Aglaonice)

*Aglaonice, an ancient Greek astronomer from the 2nd or 1st century BC, was celebrated for her precise lunar eclipse predictions. Her mastery inspired the Greek proverb: "As the moon obeys Aglaonice," signifying unwavering certainty.*

https://pypi.org/project/flatpack

## How to use Flatpack

Flatpack is currently experimental. Please refrain from using it in production environments.

Flatpack 目前仍处于实验阶段。请勿用于生产环境。

### Run in Docker (recommended)

> **SECURITY NOTICE:** Running Flatpack outside Docker allows applications to execute unrestricted Bash and Python code with your user permissions. While Docker provides some isolation, it CANNOT guarantee complete security. ONLY use Flatpack if you accept and understand these security risks.

> **Docker Engine security:** "Docker containers are, by default, quite secure; especially if you run your processes as non-privileged users inside the container. You can add an extra layer of safety by enabling AppArmor, SELinux, GRSEC, or another appropriate hardening system." ([Docker 2024](https://docs.docker.com/engine/security/#conclusions))

Learn more about [Docker Engine security](https://docs.docker.com/engine/security/).

```bash
mkdir -p ~/flatpacks
```

```bash
docker run -it \
  -p 3000:3000 \
  -p 8000:8000 \
  -v ~/flatpacks:/home/flatpackuser/flatpacks \
  flatpackai/flatpack-image bash
```

### Getting started

```bash
flatpack list
```

```bash
# Unbox from a local flatpack or .fpk file (default)
flatpack unbox my-flatpack
```

```bash
# Use the --warehouse flag for hosted flatpacks
flatpack unbox my-flatpack --warehouse
```

```bash
# Use the --use-euxo flag for debugging
flatpack build <flatpack_name>
```

```bash
# Update an unboxed flatpack
flatpack update <flatpack_name>
```

### Create a local and private flatpack

Here is how to create a local and private flatpack.

```bash
flatpack create <flatpack_name>
```

### Run the local development environment

> **SECURITY NOTICE:** This environment runs code with your permission, meaning it can connect to the Internet, install new software, which might be risky, read and change files on your computer, and slow down your computer if it does big tasks. Be careful about what code you run here.

```bash
flatpack run <flatpack_name>
```

### Share your environment online

> **SECURITY NOTICE:** Sharing your environment online exposes it to the Internet and may result in the exposure of sensitive data. You are solely responsible for managing and understanding the security risks. We are not responsible for data breaches or unauthorised access from the --share option.

```bash
export NGROK_AUTHTOKEN=<your_ngrok_token>
flatpack run <flatpack_name> --domain=<custom_ngrok_domain> --share
```

## VectorManager (HNSW)

Hierarchical Navigable Small World (HNSW) graphs excel as indexes for vector similarity searches with leading-edge results through high recall rates and rapid search capabilities.

```bash
flatpack vector add-texts "<text_1>", "<text_2>" --data-dir <data_dir>
```

```bash
flatpack vector add-pdf <pdf_filename> --data-dir <data_dir>
```

```bash
flatpack vector add-url <url> --data-dir <data_dir>
```

```bash
flatpack vector add-wikipedia "<wikipedia_page_title>" --data-dir <data_dir>
```

```bash
flatpack vector search-text "<search_query>" --data-dir <data_dir>
```

### Acknowledgements 

Our [VectorManager](https://github.com/RomlinGroup/Flatpack/blob/main/package/flatpack/flatpack/vector_manager.py) was built with:

* [hnswlib](https://github.com/nmslib/hnswlib) ([Apache 2.0](https://github.com/nmslib/hnswlib/blob/master/LICENSE)) - Fast approximate nearest neighbor search
* [sentence-transformers](https://github.com/UKPLab/sentence-transformers) ([Apache 2.0](https://github.com/UKPLab/sentence-transformers/blob/master/LICENSE)) - Sentence embeddings
* [spaCy](https://github.com/explosion/spaCy) ([MIT](https://github.com/explosion/spaCy/blob/master/LICENSE)) - Industrial-strength NLP

Special thanks to the teams behind these essential libraries.

## Model Compression

Compress [Hugging Face](https://huggingface.co) models compatible with [llama.cpp](https://github.com/ggerganov/llama.cpp) to Q4_K_M and GGUF.

Model compression cannot be run inside Docker containers.

```bash
flatpack compress <hf_model_name> --method llama.cpp --token <hf_token>
```

## Flatpack (FPK)

- [template](https://github.com/RomlinGroup/template)
  - `/app` (Next.js)
  - `app.css` (Editor)
  - `app.js` (Editor)
  - `index.html` (Editor)
  - `robotomono.woff2` (Editor)
  - `package.json` (Editor)
  - `flatpack.toml` (Flatpack)
  - `connections.json` (Flatpack)
  - `flatpack.json` (Flatpack)
  - `hooks.json` (Flatpack)
  - `source.json` (Flatpack)
  - `build.sh` (Flatpack)
  - `device.sh` (Flatpack)
  - `README.md` (Documentation)

> The Roboto Mono font ([robotomono.woff2](https://fonts.google.com/specimen/Roboto+Mono)) in our template is graciously borrowed from [https://fonts.google.com/specimen/Roboto+Mono](https://fonts.google.com/specimen/Roboto+Mono) ([Apache License 2.0](https://fonts.google.com/specimen/Roboto+Mono/license))

### Code signing

Our platform will use [RSA keys](<https://en.wikipedia.org/wiki/RSA_(cryptosystem)>) to authenticate and safeguard the integrity of flatpacks (FPK).

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
```

```bash
python bulk_compress_and_sign_fpk.py -p private_key.pem --passphrase $PASSPHRASE
```

##### Verification

```bash
python verify_signed_data_with_cli.py --public_key public_key.pem --f <fpk_path>
```

```bash
python bulk_verify_signed_data_with_cli.py --public_key public_key.pem
```

```bash
curl -o verify_fpk.sh -H 'Cache-Control: no-cache' https://raw.githubusercontent.com/RomlinGroup/Flatpack/main/utilities/compress_and_sign_fpk/verify_fpk.sh && chmod +x verify_fpk.sh && ./verify_fpk.sh <fpk_path>
```

## Inspiration (no affiliation)

> **Arduino:** "There was once a barrier between the electronics, design, and programming world and the rest of the world. Arduino has broken down that barrier." ([Arduino 2021](https://www.arduino.cc/en/about))

> **Colab:** "Anyone with an internet connection can access Colab, and use it free of charge. Millions of students use Colab every month to learn Python programming and machine learning." ([Google 2023](https://blog.google/technology/developers/google-colab-ai-coding-features/))

> **Flatpacks:** "Whilst we can find several examples of the early implementation of mass-produced products designed for flatpack delivery and self-assembly, the generally accepted opinion was that this process had its origin in Sweden." ([The Open University 2023](https://connect.open.ac.uk/money-business-and-law/flatpack-empire))

> **Game Boy:** "Perhaps the most important and best known handheld system of all time, the Game Boy featured an 8bit processor and tiny monochrome display – both rather outdated at the time. However, designer Gunpei Yokoi felt the trade-off between performance and battery life was worthwhile - and he was right." ([The Guardian 2017](https://www.theguardian.com/technology/gallery/2017/may/12/influential-handheld-games-consoles))

> **Insects:** "Insects represent more than half of the world's biodiversity and are considered to be the most evolutionarily successful group of organisms on earth." ([University of Bergen 2015](https://www.uib.no/en/news/90507/secret-success-insects))

> **ONNX:** "Many people are working on great tools, but developers are often locked in to one framework or ecosystem. ONNX is the first step in enabling more of these tools to work together by allowing them to share models." ([ONNX 2023](https://onnx.ai/about.html))

> **Raspberry Pi Foundation:** "The Raspberry Pi Foundation is a UK-based charity with the mission to enable young people to realise their full potential through the power of computing and digital technologies." ([Raspberry Pi Foundation 2023](https://www.raspberrypi.org/about/))

> **Standardisation:** "Without standards, there can be no improvement." (Taiichi Ohno)

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.