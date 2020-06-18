# Copyright 2019-2020 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Basic setup module"""
from setuptools import setup


requirements = [
    "numpy",
    "scipy",
    "numba",
    "lmfit",
    "pytest",
    "thewalrus",
]

setup(
    name="sqtom",
    version="0.1",
    description="squeezing mode tomography",
    url="https://github.com/XanaduAI/sqtom",
    author="Xanadu",
    author_email="nicolas@xanadu.ai",
    license="Apache",
    packages=["sqtom"],
    install_requires=requirements,
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type='text/markdown',
)
