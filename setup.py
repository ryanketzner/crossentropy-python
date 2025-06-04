# setup.py

from setuptools import setup, find_packages

setup(
    name="crossentropy",
    version="0.1.0",
    description="Cross‐Entropy and Importance Sampling utilities",
    author="Ryan Ketzner",
    author_email="ketzner@ucf.edu",
    packages=find_packages(),  # will include 'crossentropy' and its submodules
    install_requires=[
        "numpy",
        "scipy",
        "scikit-learn",
    ],
    python_requires=">=3.8",
)