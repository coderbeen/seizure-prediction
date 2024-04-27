from setuptools import setup

setup(
    name="edflib",
    version="0.1",
    description="A Python package for working with EDF files of CHB-MIT dataset.",
    author="Ameer Drbeeni",
    author_email="ameerasady9@email.com",
    url="https://github.com/coderbeen/edflib",
    packages=["edflib"],
    install_requires=[
        "pandas",
        "pandera",
        "numpy",
        "mne",
        "imbalanced-learn",
        "matplotlib",
    ],
)
