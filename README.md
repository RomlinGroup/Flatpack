<p align="center">
  <img src="https://romlin.com/wp-content/uploads/2023/05/flatpack_ai_logo.svg" width="100" height="100" alt="Flatpack">
</p>

# Flatpack

[![pip install flatpack](https://img.shields.io/badge/pip%20install-flatpack-5865f2)](https://pypi.org/project/flatpack/) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/romlingroup/flatpack-ai/blob/main/notebooks/flatpack_playground.ipynb)

> :warning: **DISCLAIMER:** This repository contains our research. Verify the information and do your own research (DYOR). We assume no responsibility for accuracy or completeness.

> 😱 **Regarding AI fearmongering:** ["At what specific date in the future, if the apocalypse hasn't happened, will you finally admit to being wrong?"](https://bigthink.com/pessimists-archive/ai-fear-overpopulation/) (Archie McKenzie 2023) / ["Fearmongering is a technique that has benefited many people over the ages."](https://www.youtube.com/watch?v=2ZbRKxZ2cjM) (Fred L. Smith, Jr. 2019)

## Ready-to-assemble AI

Welcome, brave explorer! We are still in stealth mode (of sorts), but we are glad you found us.

Flatpack democratises AI and ML through micro language models and model compression. Our platform enables users to train custom models with 100M to 10B parameters. We introduce flatpacks (FPKs) to integrate AI and ML into edge computing, electronic components, and robots.

> **Flatpack:** "Picture merging the Swedish ingenuity of flatpacks (hence our name) and ready-to-assemble furniture with the imaginative appeal of certain Danish snap-together toy bricks. Our method of developing micro language models is designed to be intelligently integrated into the physical world."

## Edge artificial intelligence

Edge artificial intelligence uses local devices to enhance decision-making near data sources, improving privacy, response times,  and security while reducing reliance on cloud connectivity. Benefits include decreased latency, improved scalability, and reduced energy usage.

> "Edge artificial intelligence (AI), or AI at the edge, is the implementation of artificial intelligence in an edge computing environment, which allows computations to be done close to where data is actually collected, rather than at a centralized cloud computing facility or an offsite data center." ([Red Hat 2023](https://www.redhat.com/en/topics/edge-computing/what-is-edge-ai))

## Flatpack 3.X.X (Aglaonice)

*Aglaonice, an ancient Greek astronomer from the 2nd or 1st century BC, was celebrated for her precise lunar eclipse predictions. Her mastery inspired the Greek proverb: "As the moon obeys Aglaonice," signifying unwavering certainty.*

https://pypi.org/project/flatpack

## Quick start

**NOTE:** Flatpack is currently experimental. Please refrain from using it in production environments.

```bash
# Install from PyPI
pip install flatpack
```

```bash
# Check version
flatpack version
```

```bash
# List all available flatpacks
flatpack list
```

```bash
# Unbox a flatpack of your choice
# flatpack unbox <flatpack_name> --local
flatpack unbox <flatpack_name>
```

```bash
# And last, but not least
flatpack build <flatpack_name>
```

## AgentManager

```bash
# fast_api_test.py uses microsoft/Phi-3-mini-4k-instruct-gguf (MIT)
wget https://raw.githubusercontent.com/romlingroup/flatpack/main/agents/fast_api_test.py
flatpack agents spawn fast_api_test.py
```

```bash
curl -X 'POST' \
  'http://127.0.0.1:<port>/generate-response/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "context": "<context_description>",
  "question": "<your_question>"
}'
```

```bash
flatpack agents list
```

```bash
flatpack agents terminate <pid>
```

## VectorManager (HNSW)

Hierarchical Navigable Small World (HNSW) graphs excel as indexes for vector similarity searches with leading-edge results through high recall rates and rapid search capabilities.

```bash
flatpack vector add-texts "<text_1>", "<text_2>"
```

```bash
flatpack vector add-pdf <pdf_filename>
```

```bash
flatpack vector add-wikipedia "<wikipedia_page_title>"
```

```bash
flatpack vector search-text "<search_query>"
```

## Flatpack (FPK) 📦

- [template](warehouse/template)
  - flatpack.toml
  - README.md
  - build.sh
  - custom.sh
  - device.sh
  - index.html

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
python bulk_verify_signed_data_with_cli.py --public_key public_key.pem
```

## Micro language models 🤏

[Micro language models](http://microlanguagemodels.com) with around 100M to 10B parameters, offer a compelling alternative to the deployment and scalability challenges associated with traditional large language models (LLMs). Their smaller size allows for rapid pre-training and fine-tuning, enabling organisations to adapt quickly to market changes or specific requirements. Micro language models can be deployed in edge-adjacent environments like laptops, microprocessors, or smartphones.

Overall, micro language models represent a logical step in the evolution of language models, effectively merging the capabilities of LLMs with the practical needs of real-world applications.

## Inspiration (no affiliation) ⭐

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

## Friendly notice ❤️
The [logo or symbol](https://romlin.com/wp-content/uploads/2023/05/flatpack_ai_logo.svg) associated with Flatpack is a registered trademark of [Romlin Group AB](https://romlin.com) and is protected by copyright. Please note that the logo is NOT covered by the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0) that applies to the source code in this repository. Please ask for permission if you want to use the logo for anything besides GitHub shenanigans. Thanks a million for being super awesome and respectful!