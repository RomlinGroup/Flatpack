from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="flatpack",
    version="3.1.246",
    license="Apache Software License (Apache-2.0)",
    packages=find_packages(),
    install_requires=[
        "cryptography==42.0.2",
        "fastapi==0.109.2",
        "httpx==0.26.0",
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
