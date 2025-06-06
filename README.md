# PyLeLemmatize

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
python3 setup.py develop
```

## Usage

### Command Line Invocation

#### Evaluate Merges

```sh
ll_evaluate_merges -h # get help string with the cli interface
ll_evaluate_merges -corpus_glob  './sample_data/wienocist_charter_1/wienocist_charter_1*'
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

Contributions are welcome! Please open an issue or submit a pull request on GitHub.

## License

This project is licensed under the MIT License.