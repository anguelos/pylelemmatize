from setuptools import setup, find_packages

setup(
    name='pylelematize',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
        'numpy', 'unidecode', 'fargv', 'matplotlib', 'scipy'
    ],
    author='Anguelos Nicolaou',
    author_email='anguelos.nicolaou@gmail.com',
    description='A set utilities for hadling alphabets of corpora and OCR/HTR datasets',
    long_description=open('README.md').read(),
    url='https://github.com/anguelos/pylelematize',  # Replace with your project's URL
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    entry_points={
        'console_scripts': [
            'your_command_name=your_module:your_function',  # Replace with your command, module, and function
        ],
    },
)