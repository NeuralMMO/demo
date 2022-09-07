from pdb import set_trace as T

from setuptools import find_packages, setup

REPO_URL = "https://github.com/neuralmmo/demo"

setup(
    name="nmmo",
    description="Neural MMO is a platform for multiagent intelligence research inspired by "
    "Massively Multiplayer Online (MMO) role-playing games. Documentation hosted at neuralmmo.github.io.",
    long_description_content_type="text/markdown",
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'ray-2.0.0-cp39-cp39-manylinux2014_x86_64.whl'
        'pytest-benchmark==3.4.1',
        'openskill==0.2.0-alpha.0',
        'fire==0.4.0',
        'setproctitle==1.1.10',
        'service-identity==21.1.0',
        'autobahn==19.3.3',
        'Twisted==19.2.0',
        'vec-noise==1.1.4',
        'imageio==2.8.0',
        'tqdm==4.61.1',
        'lz4==4.0.0',
        'pettingzoo',
    ],
    python_requires=">=3.8",
    license="MIT",
    author="Joseph Suarez",
    author_email="jsuarez@mit.edu",
    url=REPO_URL,
    keywords=["Neural MMO", "MMO"],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Environment :: Console",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
    ],
)

