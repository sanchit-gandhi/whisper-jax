# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os

# To use a consistent encoding
from codecs import open

from setuptools import find_packages, setup


_deps = [
    "transformers>=4.27.4",
    "flax",
    "cached-property",
]

_extras_dev_deps = [
    "black~=23.1",
    "isort>=5.5.4",
    "ruff>=0.0.241,<=0.0.259",
]

_extras_endpoint_deps = [
    "gradio>=4.9.1",
    "requests>=2.28.2",
    "yt-dlp>=2023.3.4",
]


here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# read version
with open(os.path.join(here, "whisper_jax", "__init__.py"), encoding="utf-8") as f:
    for line in f:
        if line.startswith("__version__"):
            version = line.split("=")[1].strip().strip('"')
            break
    else:
        raise RuntimeError("Unable to find version string.")

setup(
    name="whisper_jax",
    version=version,
    description="JAX implementation of OpenAI's Whisper model.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=_deps,
    extras_require={
        "dev": [_extras_dev_deps],
        "endpoint": [_extras_endpoint_deps],
    },
)
