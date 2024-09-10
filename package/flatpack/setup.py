from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="flatpack",
    version="3.7.36",
    license="Apache Software License (Apache-2.0)",
    packages=find_packages(),
    install_requires=[
        "beautifulsoup4==4.12.3",
        "croniter==3.0.3",
        "cryptography==43.0.1",
        "fastapi==0.114.0",
        "hnswlib==0.8.0",
        "httpx==0.27.2",
        "huggingface-hub==0.24.6",
        "ngrok==1.4.0",
        "nltk==3.9.1",
        "prettytable==3.11.0",
        "psutil==6.0.0",
        "pydantic==2.9.0",
        "pypdf==4.3.1",
        "python-multipart==0.0.9",
        "requests==2.32.3",
        "sentence-transformers==3.0.1",
        "toml==0.10.2",
        "torch==2.4.1",
        "uvicorn==0.30.6",
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
