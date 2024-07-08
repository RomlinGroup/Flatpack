from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="flatpack",
    version="3.5.83",
    license="Apache Software License (Apache-2.0)",
    packages=find_packages(),
    install_requires=[
        "beautifulsoup4==4.12.3",
        "fastapi==0.111.0",
        "hnswlib==0.8.0",
        "httpx==0.27.0",
        "huggingface-hub==0.23.4",
        "ngrok==1.3.0",
        "nltk==3.8.1",
        "psutil==5.9.5",
        "pypdf==4.2.0",
        "python-multipart==0.0.9",
        "requests==2.32.3",
        "sentence-transformers==3.0.1",
        "toml==0.10.2",
        "torch==2.3.1",
        "uvicorn==0.30.1"
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
