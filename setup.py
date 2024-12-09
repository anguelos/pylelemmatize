from setuptools import setup, find_packages

setup(
    name='pylelemmatize',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
        'numpy', 'unidecode', 'fargv', 'matplotlib', 'scipy', 'tqdm', 'networkx'
    ],
    author='Anguelos Nicolaou',
    author_email='anguelos.nicolaou@gmail.com',
    description='A set utilities for hadling alphabets of corpora and OCR/HTR datasets',
    long_description=open('README.md').read(),
    url='https://github.com/anguelos/pylelemmatize',  # Replace with your project's URL
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    entry_points={
        'console_scripts': [
            'll_extract_corpus_alphabet=pylelemmatize:main_alphabet_extract_corpus_alphabet',
            'll_test_corpus_on_alphabets=pylelemmatize:main_map_test_corpus_on_alphabets',
            'll_evaluate_merges=pylelemmatize:main_alphabet_evaluate_merges',
        ],
    },
)