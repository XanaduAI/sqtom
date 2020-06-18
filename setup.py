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
from setuptools import setup, find_packages


with open("sqtom/_version.py") as f:
    version = f.readlines()[-1].split()[-1].strip("\"'")


requirements = [
    "numpy",
    "scipy",
    "numba",
    "lmfit",
    "thewalrus",
]


classifiers = [
    "Development Status :: 4 - Beta",
    "Environment :: Console",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: Apache Software License",
    "Natural Language :: English",
    "Operating System :: POSIX",
    "Operating System :: MacOS :: MacOS X",
    "Operating System :: POSIX :: Linux",
    "Operating System :: Microsoft :: Windows",
    "Programming Language :: Python",
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3 :: Only',
    "Topic :: Scientific/Engineering :: Physics"
]


setup(
    name="sqtom",
    version=version,
    description="squeezing mode tomography",
    url="https://github.com/XanaduAI/sqtom",
    author="Xanadu",
    author_email="nicolas@xanadu.ai",
    license="Apache License 2.0",
    packages=find_packages(where="."),
    install_requires=requirements,
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type='text/markdown',
    classifiers=classifiers,
)
