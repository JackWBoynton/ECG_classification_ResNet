import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ecgaugmentation", # Replace with your own username
    version="0.0.1",
    author="Jack Boynton",
    author_email="jwboynto@uvm.edu",
    description="ECG augmentation ",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jackwboynton/timeseries_resnet/ecg-augmentation",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)