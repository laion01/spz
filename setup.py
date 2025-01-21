import pathlib

from setuptools import find_packages
from skbuild import setup

HERE = pathlib.Path(__file__).parent

setup(
    name="pyspz",
    version="1.0.0",
    description="Python bindings for SPZ Gaussian Splat library",
    long_description=(HERE / "README.md").read_text(),
    long_description_content_type="text/markdown",
    author="Denis Avvakumov",
    author_email="denisavvakumov@gmail.com",
    url="https://github.com/404-Repo/pyspz",
    packages=find_packages(),
    install_requires=[],
    cmake_install_dir="pyspz",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries",
    ],
    license="MIT",
    python_requires=">=3.10",
    include_package_data=True,
    zip_safe=False,
    cmake_args=[
        "-DCMAKE_BUILD_TYPE=Release",
    ],
)
