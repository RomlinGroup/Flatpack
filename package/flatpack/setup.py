from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="flatpack",
    version="3.4.72",
    license="Apache Software License (Apache-2.0)",
    packages=find_packages(),
    install_requires=[
        "beautifulsoup4==4.12.3",
        "cryptography==42.0.6",
        "faiss-cpu==1.8.0",
        "fastapi==0.110.2",
        "hnswlib==0.8.0",
        "httpx==0.27.0",
        "huggingface-hub==0.23.0",
        "llama-cpp-python==0.2.75",
        "ngrok==1.2.0",
        "nltk==3.8.1",
        "olefile==0.47",
        "psutil==5.9.5",
        "pydantic==2.7.1",
        "pypdf==4.2.0",
        "requests==2.31.0",
        "sentence-transformers==2.7.0",
        "toml==0.10.2",
        "uvicorn==0.29.0"
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
