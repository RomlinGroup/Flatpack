<div align="center">
  <img src="https://raw.githubusercontent.com/romlingroup/flatpack-ai/main/client/static/images/flatpack_ai_logo.svg" width="200" height="200" alt="Flatpack AI">
</div>

# flatpack.ai 🤖

[![Rust](https://github.com/romlingroup/flatpack-ai/actions/workflows/rust.yml/badge.svg)](https://github.com/romlingroup/flatpack-ai/actions/workflows/rust.yml)

>:warning: **DISCLAIMER:** This repository contains our research. Verify the information and do your own research (DYOR). We assume no responsibility for accuracy or completeness.

>😱 **Regarding AI fearmongering:** [All doom prophets should be required to answer this question: "At what specific date in the future, if the apocalypse hasn't happened, will you finally admit to being wrong?"](https://bigthink.com/pessimists-archive/ai-fear-overpopulation/) (Archie McKenzie 2023) / [How politicians and groups can profit from fearmongering](https://www.youtube.com/watch?v=2ZbRKxZ2cjM) (Fred L. Smith, Jr. 2019)


flatpack.ai will democratize AI by providing a modular and open platform for anyone to train their AI models from scratch with cutting-edge technology accessible to all. What flatpacks did for the furniture industry, we will do for the AI industry.

## The flatpack.ai client (Rust) 🦀

To use the client, follow these steps:

1. git clone https://github.com/romlingroup/flatpack-ai.git
2. Install Rust with https://www.rust-lang.org/tools/install
3. To parse a file, run: `cargo run -- parse /path/to/your/file.toml`
4. To start the server, run: `cargo run -- run-server`
5. Visit http://localhost:8080

## Container (Podman) 🦭

1. Install [Podman](https://podman.io/) ([Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0))
2. `podman build -t nanogpt-shakespeare -f Containerfile .`
3. `podman run -it nanogpt-shakespeare:latest`

## Commercial use allowed (no affiliation) 📈

We constantly search for datasets and models suitable for future deployment as [flatpacks](https://github.com/romlingroup/flatpack-ai/tree/main/warehouse) (coming soon). Therefore, if you know of any high-quality datasets or models with commercially viable licenses, we would appreciate it if you submitted them via a pull request.

| Name                                                                                              | Type    | License                                                           |
|:--------------------------------------------------------------------------------------------------|:--------|:------------------------------------------------------------------|
| [databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k)           | Dataset | [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/)   |
| [Cerebras-GPT-13B](https://huggingface.co/cerebras/Cerebras-GPT-13B)                              | Model   | [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0) |
| [mpt-7b](https://huggingface.co/mosaicml/mpt-7b)                                                  | Model   | [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0) |
| [mpt-7b-instruct](https://huggingface.co/mosaicml/mpt-7b-instruct)                                | Model   | [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/)   |
| [mpt-7b-storywriter](https://huggingface.co/mosaicml/mpt-7b-storywriter)                          | Model   | [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0) |
| [open_llama_3b_350bt_preview](https://huggingface.co/openlm-research/open_llama_3b_350bt_preview) | Model   | [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0) |
| [open_llama_7b_400bt_preview](https://huggingface.co/openlm-research/open_llama_7b_400bt_preview) | Model   | [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0) |
| [pythia-12b](https://huggingface.co/EleutherAI/pythia-12b)                                        | Model   | [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0) |
| [stablelm-base-alpha-3b](https://huggingface.co/stabilityai/stablelm-base-alpha-3b)               | Model   | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)   |
| [stablelm-base-alpha-7b](https://huggingface.co/stabilityai/stablelm-base-alpha-7b)               | Model   | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)   |

### Friendly notice ❤️

The flatpack.ai logo belongs to [Romlin Group AB](https://romlin.com) and is protected by copyright. Please note that the logo is NOT covered by the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0) that applies to the source code in this repository. If you want to use the logo for anything besides GitHub shenanigans, please ask for permission first. Thanks a million for being super awesome and respectful!