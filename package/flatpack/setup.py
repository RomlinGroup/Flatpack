from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="flatpack",
    version="3.7.6",
    license="Apache Software License (Apache-2.0)",
    packages=find_packages(),
    install_requires=[
        "beautifulsoup4==4.12.3",
        "croniter==3.0.3",
        "cryptography==43.0.0",
        "fastapi==0.112.0",
        "hnswlib==0.8.0",
        "httpx==0.27.0",
        "huggingface-hub==0.24.5",
        "ngrok==1.4.0",
        "nltk==3.8.1",
        "prettytable==3.10.2",
        "psutil==6.0.0",
        "pydantic==2.8.2",
        "pypdf==4.3.1",
        "python-multipart==0.0.9",
        "requests==2.32.3",
        "sentence-transformers==3.0.1",
        "toml==0.10.2",
        "torch==2.4.0",
        "uvicorn==0.30.5",
        "zstandard==0.23.0"
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
