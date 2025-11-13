"""
Setup script for MAVERICK package
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="maverick-ai",
    version="0.6.0",
    author="Koppi02",
    description="Hierarchikus hirdetés klasszifikáló AI rendszer új kategória felismeréssel",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Koppi02/AI-Agent",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "maverick-train=scripts.train:main",
            "maverick-predict=scripts.predict:main",
        ],
    },
)
