from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="flatpack",
    version="3.1.277",
    license="Apache Software License (Apache-2.0)",
    packages=find_packages(),
    install_requires=[
        "chromadb==0.4.24",
        "cryptography==42.0.5",
        "fastapi==0.110.0",
        "httpx==0.27.0",
        "ngrok==1.0.0",
        "toml==0.10.2",
        "uvicorn==0.27.1"
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
