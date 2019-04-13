import os
from setuptools import setup, find_packages


def read(file_name):
    return open(os.path.join(os.path.dirname(__file__), file_name)).read()

setup(
    name='App Classifier',
    version='0.2',
    url='https://github.com/trevorxander/AppClassifier',
    license='',
    author='Trevor Xander',
    author_email='trevorcolexander@gmail.com',
    install_requires=['numpy', 'pandas', 'scikit-learn', 'polyglot', 'PyICU', 'pycld2'],
    packages=find_packages()
)
