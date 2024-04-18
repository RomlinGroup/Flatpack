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

## Flatpack 3.X.X (Aglaonice)

*Aglaonice, an ancient Greek astronomer from the 2nd or 1st century BC, was celebrated for her precise lunar eclipse predictions. Her mastery inspired the Greek proverb: "As the moon obeys Aglaonice," signifying unwavering certainty.*

https://pypi.org/project/flatpack

## Quick start

**NOTE:** Flatpack is currently experimental. Please refrain from using it in production environments.

```bash
# Colab: /content
# Linux: /home/<username>/flatpacks
# macOS: /Users/<username>/flatpacks
# Windows: Use WSL2 and see Linux path

# Install from PyPI
pip install flatpack==3.3.0
```

```bash
# List all available flatpacks
flatpack list
```

```bash
# Unbox a flatpack of your choice
flatpack unbox moondream
```

```bash
# And last, but not least
flatpack build moondream
```

## Flatpack (FPK) 📦

- [template](warehouse/template)
  - /app
  - flatpack.toml
  - DATASET.md
  - README.md
  - build.sh
  - device.sh

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

## DATASET.md 📖

[DATASET.md](https://github.com/romlingroup/flatpack-ai/blob/main/datasets/DATASET.md) is our meticulously crafted Markdown template (under development), established to act as a standard for encapsulating the essentials of any dataset. By streamlining the documentation process, this template ensures that all relevant details about the dataset are easily accessible, facilitating seamless collaboration and utilization.

Markdown is ideal for documenting datasets as it is lightweight and easy to archive. Its format helps keep data and documentation in sync, essential for research integrity and reproducibility. Markdown files and datasets can also be version-controlled, ensuring a cohesive historical record.

Once completed, it will be a mandatory component in all flatpacks.

## Micro language models 🤏

[Micro language models](http://microlanguagemodels.com) with around 100M to 10B parameters, offer a compelling alternative to the deployment and scalability challenges associated with traditional large language models (LLMs). Their smaller size allows for rapid pre-training and fine-tuning, enabling organisations to adapt quickly to market changes or specific requirements. Micro language models can be deployed in edge-adjacent environments like laptops, microprocessors, or smartphones, benefiting from [edge computing](https://en.wikipedia.org/wiki/Edge_computing) (computation and storage closer to data sources) to facilitate low-latency and privacy-conscious applications. These characteristics make micro language models broadly accessible, energy-efficient, and specialized, even to smaller teams or individual developers.

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

## Navformer 🧭

We are developing [Navformer](http://navformer.com), an AI-powered navigation system for robots that avoids obstacles and executes commands efficiently. It will be a standard component in Flatpack.

* **CPU-friendly:** Optimised for CPU usage, ensuring compatibility and efficiency across various edge computing without requiring specialised hardware.
* **Monocular navigation:** Utilises a single-lens system for spatial awareness, providing a compact and cost-effective solution for 3D environment mapping.
* **Real-time inference:** Capable of processing and reacting to environmental data in real-time, crucial for dynamic and unpredictable settings.
* **Transformer-based:** Incorporates the latest in transformer neural network technology, enabling interpretation of complex spatial data and superior understanding.

## Hello Higgs 🤖

Our conceptual robot, [Higgs](http://hellohiggs.com), is named after the narrator in ["Erewhon"](https://en.wikipedia.org/wiki/Erewhon) (1872) by Samuel Butler.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Friendly notice ❤️
The [logo or symbol](https://romlin.com/wp-content/uploads/2023/05/flatpack_ai_logo.svg) associated with Flatpack is a registered trademark of [Romlin Group AB](https://romlin.com) and is protected by copyright. Please note that the logo is NOT covered by the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0) that applies to the source code in this repository. Please ask for permission if you want to use the logo for anything besides GitHub shenanigans. Thanks a million for being super awesome and respectful!