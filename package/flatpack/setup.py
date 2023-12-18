from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="flatpack",
    version="3.1.23",
    license="Apache Software License (Apache-2.0)",
    packages=find_packages(),
    install_requires=[
        "cryptography==41.0.7",
        "gradio==4.8.0",
        "httpx==0.25.2",
        "numpy==1.26.2",
        "toml==0.10.2",
        "transformers==4.36.0"
    ],
    author="Romlin Group AB",
    author_email="hello@romlin.com",
    description="Ready-to-assemble AI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    entry_points={
        "console_scripts": [
            "flatpack=flatpack.main:main"
        ],
    }
)
