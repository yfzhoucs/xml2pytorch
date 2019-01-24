import setuptools

with open("./README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="xml2pytorch",
    version="0.0.9",
    author="Yifan Zhou",
    author_email="yfzhou.cs@gmail.com",
    description="Using xml to define pytorch neural networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yfzhoucs/xml2pytorch",
    packages=setuptools.find_packages(),
    install_requires = [
        "torch>=0.4.1",
        "numpy>=1.15.1"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)