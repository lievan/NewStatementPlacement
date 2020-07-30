from setuptools import setup, find_packages

setup(
    name='new-statement-placement',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'transformers>=3.0.2',
        'torch>=1.2.0',
        'numpy',
        'pandas',
    ],
    url='https://github.com/lievan/new-statement-placement',
    license='Attribution-NonCommercial-ShareAlike 3.0',
    author='Evan Li',
    author_email='el3078@columbia.edu',
    description='Recommends placement of a new statement on an argument map using BERT'
)
