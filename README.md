# PyLeLemmatize

[![PyPI](https://img.shields.io/pypi/v/pylelemmatize.svg)](https://pypi.org/project/pylelemmatize/)
[![Python](https://img.shields.io/pypi/pyversions/pylelemmatize.svg)](https://pypi.org/project/pylelemmatize/)
[![Build](https://github.com/anguelos/pylelemmatize/actions/workflows/tests.yml/badge.svg)](https://github.com/anguelos/pylelemmatize/actions/workflows/tests.yml)
[![Docs](https://readthedocs.org/projects/pylelemmatize/badge/?version=latest)](https://pylelemmatize.readthedocs.io/en/latest/)
[![License: MIT](https://img.shields.io/github/license/anguelos/pylelemmatize.svg)](https://github.com/anguelos/pylelemmatize/blob/main/LICENSE)

A fast, modular lemmatization toolkit for Python.



PyLeLemmatize is a Python package for lemmatizing text. It provides a simple and efficient way to reduce large characters to simpler ones.

## Installation

### Install from GitHub with pip

To install PyLemmatize directly from GitHub using pip, run the following command:

```sh
pip install git+https://github.com/yourusername/pylelemmatize.git
```

### Install from GitHub with code

To install PyLemmatize from the source code, follow these steps:

1. Clone the repository:
2. Navigate to the project directory:
3. Install the package

```sh
git clone https://github.com/yourusername/pylelemmatize.git
cd pylelemmatize
pip install -e ./  
# If you dont want a development install, do pip install ./
```

## Usage

### Command Line Invocation

#### Evaluate Merges

```sh
ll_evaluate_merges -h # get help string with the cli interface
ll_evaluate_merges -corpus_glob  './sample_data/wienocist_charter_1/wienocist_charter_1*'
```

Attention the merge CER is not symetric at all!
```
# The following gives a CER of 0.0591
ll_evaluate_merges -corpus_glob  './sample_data/wienocist_charter_1/wienocist_charter_1*' -merges '[("I", "J"), ("i", "j")]'
# While the following gives a CER of 0.0007
ll_evaluate_merges -corpus_glob  './sample_data/wienocist_charter_1/wienocist_charter_1*' -merges '[("J", "I"), ("j", "i")]'
```

#### Extract corpus alphabet
```sh
ll_extract_corpus_alphabet -h # get help string with the cli interface
ll_extract_corpus_alphabet -corpus_glob './sample_data/wienocist_charter_1/wienocist_charter_1*'
```

#### Test corpus on alphabets
```sh
ll_test_corpus_on_alphabets -h # get help string with the cli interface
ll_test_corpus_on_alphabets -corpus_glob './sample_data/wienocist_charter_1/wienocist_charter_1*' -alphabets 'bmp_mufi,ascii,mes1,iso8859_2' -verbose
```

## Contributing

Contributions are welcome!

## License

This project is licensed under the MIT License.