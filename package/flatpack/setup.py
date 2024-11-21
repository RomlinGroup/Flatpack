from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="flatpack",
    version="3.10.36",
    license="Apache Software License (Apache-2.0)",
    packages=find_packages(),
    python_requires=">=3.10.0",
    install_requires=[
        "beautifulsoup4==4.12.3",
        "croniter==3.0.3",
        "cryptography==43.0.1",
        "fastapi==0.115.2",
        "hnswlib==0.8.0",
        "httpx==0.27.2",
        "huggingface-hub==0.25.2",
        "libcst==1.5.0",
        "itsdangerous==2.2.0",
        "ngrok==1.4.0",
        "prettytable==3.11.0",
        "psutil==6.1.0",
        "pydantic==2.9.2",
        "pypdf==5.0.1",
        "python-multipart==0.0.12",
        "requests==2.32.3",
        "rich==13.9.2",
        "sentence-transformers==3.2.0",
        "spacy==3.8.2",
        "toml==0.10.2",
        "uvicorn==0.32.0",
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
