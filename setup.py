import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="dexter-ai-proctor",
    version="0.3",
    author="Vardan Agarwal",
    author_email="vardanagarwal16@gmail.com",
    description="A package for AI proctoring of dexter learning.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dexter-learning/dexter-ai-proctor",
    project_urls={
        "Bug Tracker": "https://github.com/dexter-learning/dexter-ai-proctor/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.7",
    include_package_data=True
    )
