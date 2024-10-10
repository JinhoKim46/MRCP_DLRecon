"""
Copyright (c) Facebook, Inc. and its affiliates.
Copyright (c) 2024 Jinho Kim (jinho.kim@fau.de)

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import re

from setuptools import find_packages, setup

# get version string from module
with open(os.path.join(os.path.dirname(__file__), "dlrecon/__init__.py"), "r") as f:
    readval = re.search(r"__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
    if readval is None:
        raise RuntimeError("Version not found.")
    version = readval.group(1)
    print("-- Building version " + version)

with open("README.md", encoding="utf8") as f:
    readme = f.read()

install_requires = [
    "torch==2.1.2",
    "torchmetrics",
    "torchvision",
    "pytorch_lightning==2.2.5",
    "lightning[pytorch-extra]",
    "numpy<1.25",
    "scikit_image>=0.16.2",
    "pandas==1.5.3",
    "h5py>=3.10.0",
    "PyYAML>=5.3.1",
]
dependency_links = ["https://download.pytorch.org/whl/cu118"]

setup(
    name="dlrecon",
    author="Jinho Kim",
    author_email="jinho.kim@fau.de",
    version=version,
    license="MIT",
    description="Deep Learning-based reconstruction for MRCP",
    long_description_content_type="text/markdown",
    long_description=readme,
    python_requires="==3.10.*",
    setup_requires=["wheel"],
    install_requires=install_requires,
    dependency_links=dependency_links,
    packages=["dlrecon"],
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)
