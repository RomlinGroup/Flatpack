<p align="center">
  <img src="https://romlin.com/wp-content/uploads/2023/05/flatpack_ai_logo.svg" width="100" height="100" alt="Flatpack">
</p>

# Flatpack

**NOTE:** Flatpack is currently experimental. Please refrain from using it in production environments.

**注意:** Flatpack 目前仍处于实验阶段。请勿用于生产环境。

## Ready-to-assemble AI

At Flatpack, our mission is clear: we are committed to building trust in AI.

Flatpack democratises AI and ML through micro language models and model compression. Our platform enables users to train
custom language models with 100M to 10B parameters. We introduce flatpacks (FPKs) to integrate AI and ML into edge
computing, electronic components, and robots.

在 Flatpack，我们的使命十分明确: 致力于建立对 AI 的信任。

Flatpack 通过微语言模型和模型压缩普及 AI 和 ML。我们的平台使用户能够训练具有 100M 到 10B 个参数的自定义语言模型。我们引入了
flatpacks (FPK)，将 AI 和 ML 集成到边缘计算、电子元件和机器人中。

## Flatpack 3.X.X (Aglaonice)

*Aglaonice, an ancient Greek astronomer from the 2nd or 1st century BC, was celebrated for her precise lunar eclipse
predictions. Her mastery inspired the Greek proverb: "As the moon obeys Aglaonice," signifying unwavering certainty.*

`3.11.0` (2024-11-24)\
*Major core architecture refactoring for performance.*

`3.10.0` (2024-10-21)\
*Added dark mode support for a better user experience.*

`3.9.0` (2024-10-04)\
*Added [Monaco Editor](https://github.com/microsoft/monaco-editor) and custom hooks support.*

`3.8.0` (2024-09-24)\
*Improved package setup and deprecated agent spawning.*

`3.7.0` (2024-09-04)\
*Added cron and manual scheduling for builds.*

`3.6.0` (2024-07-28)\
*Introduced SQLite database support for each flatpack.*

`3.5.0` (2024-05-27)\
*Initial support for model compression in GGUF using [llama.cpp](https://github.com/ggerganov/llama.cpp).*

`3.4.0` (2024-04-30)\
*Added support for spawning agents with micro APIs.*

`3.3.0` (2024-04-16)\
*Added a vector database for storing and querying embeddings.*

`3.2.0` (2024-03-09)\
*Added support for unboxing local flatpacks using --local.*

`3.1.0` (2023-12-11)\
*Introduced a local web interface for a better user experience.*

`3.0.0` (2023-10-20)\
*Moved to a predictable and structured release strategy.*

## Moving to versioning structure: 3.0.0

Our previous releases were a mix of minor tweaks and significant shifts, making it challenging to anticipate the nature
of changes. With the introduction of version 3.0.0, we are embracing a predictable and structured release strategy:

- **Major versions** (`X.0.0`): These signify significant changes or updates.
- **Minor versions** (`3.X.0`): Introduce new features without breaking compatibility.
- **Patch versions** (`3.0.X`): Address bug fixes and minor refinements.

This move is not merely about [semantic versioning](https://semver.org/). It is a pledge for clear communication and
trust with our users. We invite you to explore our new release strategy and appreciate your patience as we evolve.

[Explore the Project on GitHub](https://github.com/RomlinGroup/Flatpack)

## License

This project is released under [Apache-2.0](https://github.com/RomlinGroup/Flatpack/blob/main/LICENSE).

## install_requires

**DISCLAIMER:** This information is only a technical reference, not an endorsement or legal advice. Before using any
software for commercial purposes, perform compatibility checks and seek legal advice. You are responsible for ensuring
compliance with licensing requirements.

Check out the [JLA - Compatibility Checker](https://joinup.ec.europa.eu/collection/eupl/solution/joinup-licensing-assistant/jla-compatibility-checker) (European Commission 2024).

- **[beautifulsoup4](https://pypi.org/project/beautifulsoup4/)**\
  MIT License (MIT License) ([LICENSE](https://pypi.org/project/beautifulsoup4/))

- **[croniter](https://pypi.org/project/croniter/)**\
  MIT License (MIT License) ([LICENSE](https://github.com/kiorky/croniter/blob/master/LICENSE))

- **[fastapi](https://pypi.org/project/fastapi/)**\
  MIT License ([LICENSE](https://github.com/tiangolo/fastapi/blob/master/LICENSE))

- **[hnswlib](https://pypi.org/project/hnswlib/)**\
  Apache-2.0 license ([LICENSE](https://github.com/nmslib/hnswlib/blob/master/LICENSE))

- **[httpx](https://pypi.org/project/httpx/)**\
  BSD License ([LICENSE](https://github.com/encode/httpx/blob/master/LICENSE.md))

- **[huggingface-hub](https://pypi.org/project/huggingface-hub/)**\
  Apache Software License (Apache) ([LICENSE](https://github.com/huggingface/huggingface_hub/blob/main/LICENSE))

- **[itsdangerous](https://pypi.org/project/itsdangerous/)**\
  BSD License ([LICENSE](https://github.com/pallets/itsdangerous/blob/main/LICENSE.txt))

- **[ngrok](https://pypi.org/project/ngrok/)**\
  MIT License (MIT OR Apache-2.0) ([LICENSE](https://github.com/ngrok/ngrok-python/blob/main/LICENSE-APACHE))

- **[prettytable](https://pypi.org/project/prettytable/)**\
  BSD License (BSD (3 clause)) ([LICENSE](https://github.com/jazzband/prettytable/blob/main/LICENSE))

- **[psutil](https://pypi.org/project/psutil/)**\
  BSD License (BSD-3-Clause) ([LICENSE](https://github.com/giampaolo/psutil/blob/master/LICENSE))

- **[pydantic](https://pypi.org/project/pydantic/)**\
  MIT License ([LICENSE](https://github.com/pydantic/pydantic/blob/main/LICENSE))

- **[pypdf](https://pypi.org/project/pypdf/)**\
  BSD License ([LICENSE](https://github.com/py-pdf/pypdf/blob/main/LICENSE))

- **[python-multipart](https://pypi.org/project/python-multipart/)**\
  Apache Software License ([LICENSE](https://github.com/Kludex/python-multipart/blob/master/LICENSE.txt))

- **[requests](https://pypi.org/project/requests/)**\
  Apache Software License (Apache 2.0) ([LICENSE](https://github.com/psf/requests/blob/main/LICENSE))

- **[rich](https://pypi.org/project/rich/)**\
  MIT License (MIT) ([LICENSE](https://github.com/Textualize/rich/blob/master/LICENSE))

- **[sentence-transformers](https://pypi.org/project/sentence-transformers/)**\
  Apache Software License (Apache License 2.0) ([LICENSE](https://github.com/UKPLab/sentence-transformers/blob/master/LICENSE))

- **[spacy](https://pypi.org/project/spacy/)**\
  MIT License (MIT) ([LICENSE](https://github.com/explosion/spaCy/blob/master/LICENSE))

- **[toml](https://pypi.org/project/toml/)**\
  MIT License (MIT) ([LICENSE](https://github.com/uiri/toml/blob/master/LICENSE))

- **[uvicorn](https://pypi.org/project/uvicorn/)**\
  BSD License ([LICENSE](https://github.com/encode/uvicorn/blob/master/LICENSE.md))

- **[zstandard](https://pypi.org/project/zstandard/)**\
  BSD License (BSD) ([LICENSE](https://github.com/indygreg/python-zstandard/blob/main/LICENSE))

Last updated: 2024-12-16