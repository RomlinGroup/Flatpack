from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="flatpack",
    version="3.1.50",
    license="Apache Software License (Apache-2.0)",
    packages=find_packages(),
    install_requires=[
        "cryptography==41.0.7",
        "fastapi==0.105.0",
        "httpx==0.25.2",
        "Pillow==10.0.1",
        "toml==0.10.2",
        "transformers==4.36.2",
        "uvicorn==0.24.0.post1"
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
