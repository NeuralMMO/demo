from pdb import set_trace as T

from setuptools import find_packages, setup

REPO_URL = "https://github.com/neuralmmo/demo"

setup(
    name="nmmo-demo",
    description="Neural MMO is a platform for multiagent intelligence research inspired by "
    "Massively Multiplayer Online (MMO) role-playing games. Documentation hosted at neuralmmo.github.io.",
    long_description_content_type="text/markdown",
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'ray[all]==2.0.0',
        'opencv-python==3.4.17.63',
        'nmmo==1.6.0.2',
        'openskill==2.4.0',
        'magent==0.2.4',
        'pufferlib',
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

