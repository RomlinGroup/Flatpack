from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="flatpack",
    version="3.11.12",
    license="Apache Software License (Apache-2.0)",
    packages=find_packages(),
    python_requires=">=3.10.0",
    install_requires=[
        "beautifulsoup4==4.12.3",
        "croniter==5.0.1",
        "cryptography==43.0.3",
        "fastapi==0.115.5",
        "hnswlib==0.8.0",
        "httpx==0.27.2",
        "huggingface-hub==0.26.2",
        "itsdangerous==2.2.0",
        "ngrok==1.4.0",
        "prettytable==3.12.0",
        "psutil==6.1.0",
        "pydantic==2.10.1",
        "pypdf==5.1.0",
        "python-multipart==0.0.17",
        "requests==2.32.3",
        "rich==13.9.4",
        "sentence-transformers==3.3.1",
        "spacy==3.8.2",
        "toml==0.10.2",
        "uvicorn==0.32.1",
        "zstandard==0.23.0"
    ],
    author="Romlin Group AB",
    author_email="hello@romlin.com",
    description="Ready-to-assemble AI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/RomlinGroup/Flatpack",
    entry_points={
        "console_scripts": [
            "flatpack=flatpack.main:main"
        ]
    }
)
