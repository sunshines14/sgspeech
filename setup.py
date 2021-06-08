import setuptools

requirements = [
    "tensorflow-datasets>=4.1.0",
    "tensorflow-addons>=0.10.0",
    "tensorflow-io>=0.16.0",
    "setuptools>=47.1.1",
    "librosa>=0.8.0",
    "soundfile>=0.10.3",
    "PyYAML>=5.3.1",
    "matplotlib>=3.2.1",
    "sox>=1.4.1",
    "tqdm>=4.54.1",
    "colorama>=0.4.4",
    "nlpaug>=1.1.1",
    "nltk>=3.5",
    "sentencepiece>=0.1.94"
]

setuptools.setup(
    name="SGSpeech",
    version="0.0.1",
    author="Hosung Park",
    packages=setuptools.find_packages(include=["SGSpeech*"]),
    install_requires=requirements,

    python_requires='>=3.6',
)
