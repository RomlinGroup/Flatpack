from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="flatpack",
    version="3.9.28",
    license="Apache Software License (Apache-2.0)",
    packages=find_packages(),
    python_requires=">=3.10.0",
    install_requires=[
        "beautifulsoup4==4.12.3",
        "croniter==3.0.3",
        "cryptography==43.0.1",
        "fastapi==0.115.0",
        "hnswlib==0.8.0",
        "httpx==0.27.2",
        "huggingface-hub==0.25.1",
        "itsdangerous==2.2.0",
        "ngrok==1.4.0",
        "prettytable==3.11.0",
        "pydantic==2.9.2",
        "pypdf==5.0.1",
        "python-multipart==0.0.12",
        "requests==2.32.3",
        "rich==13.9.2",
        "sentence-transformers==3.1.1",
        "spacy==3.7.5",
        "toml==0.10.2",
        "uvicorn==0.31.0",
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
