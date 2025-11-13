

<p align="center">
  <picture>
    <!-- Used by Sphinx (relative path inside docs/) -->
    <source srcset="../../../docs/_static/images/logo.png">
    <!-- Used by GitHub / PyPI -->
    <img alt="PyLemmatize" src="https://raw.githubusercontent.com/anguelos/pylelemmatize/main/docs/_static/images/logo.png" width="640">
  </picture>
</p>


# Getting started

[![PyPI](https://img.shields.io/pypi/v/pylelemmatize.svg)](https://pypi.org/project/pylelemmatize/)
[![Python](https://img.shields.io/pypi/pyversions/pylelemmatize.svg)](https://pypi.org/project/pylelemmatize/)
[![Build](https://github.com/anguelos/pylelemmatize/actions/workflows/tests.yml/badge.svg)](https://github.com/anguelos/pylelemmatize/actions/workflows/tests.yml)
[![Docs](https://readthedocs.org/projects/pylelemmatize/badge/?version=latest)](https://pylelemmatize.readthedocs.io/en/latest/)
[![License: MIT](https://img.shields.io/github/license/anguelos/pylelemmatize.svg)](https://github.com/anguelos/pylelemmatize/blob/main/LICENSE)

A framework for assisting transliterations and character-sets in python.




PyLeLemmatize is a Python package for lemmatizing characters. It provides a simple and efficient way to reduce large character sets to simpler ones.

## Installation

### Install from pypi

To install PyLemmatize from Pypi:

```sh
pip install pylelemmatize
```
for installation for coding, look at [development](### Development Installation)

## Python Usage

### Simple letter lemmatization
```python
from pylelemmatize import charsets, llemmatize

greek_poly_string = "Καὶ ὅτε ἤνοιξεν τὴν σφραγῖδα τὴν ἑβδόμην, ἐγένετο σιγὴ ἐν τῷ οὐρανῷ ὡς ἡμιώριον."

print(f"Polytonic   : {greek_poly_string}")
print(f"Modern Greek: {llemmatize(greek_poly_string, charsets.iso_8859_7)}")
print(f"ASCII       : {llemmatize(greek_poly_string, charsets.ascii)}")
```
Output:
```console
Polytonic   : Καὶ ὅτε ἤνοιξεν τὴν σφραγῖδα τὴν ἑβδόμην, ἐγένετο σιγὴ ἐν τῷ οὐρανῷ ὡς ἡμιώριον.
Modern Greek: Καί ότε ήνοιξεν τήν σφραγίδα τήν έβδόμην, έγένετο σιγή έν τώ ούρανώ ώς ήμιώριον.
ASCII       : Kai ote enoixen ten spragida ten ebdomen, egeneto sige en to ourano os emiorion.
```

### Efficient letter lemmatization

Creating automoatic llemmatizers is expencive O(|input_alphabet|x|output_alphabet|)
Once they are created they are equally fast regardless of of their sizes.
The following IPython codesnipet demonstrates the cost of creating vs applying llemmatizers.
```python
from pylelemmatize import charsets, llemmatizer

greek_poly_string = "Καὶ ὅτε ἤνοιξεν τὴν σφραγῖδα τὴν ἑβδόμην, ἐγένετο σιγὴ ἐν τῷ οὐρανῷ ὡς ἡμιώριον."

print("Creating autoaligned llemmatizers O(|src_alphabet|x|dst_alphabet|)")
print("Medium llemmatizer: |34|x|186|")
%timeit polytonic2modern_greek = llemmatizer(greek_poly_string, charsets.iso_8859_7)
polytonic2modern_greek = llemmatizer(greek_poly_string, charsets.iso_8859_7)

print("Large llemmatizer: |100|x|3549|")
%timeit mes2ascii = llemmatizer(charsets.mes3a, charsets.ascii)
mes2ascii = llemmatizer(charsets.mes3a, charsets.ascii)

print("\nApplying the medium and large llemmatizers on strings:")
for inp_str in [greek_poly_string, greek_poly_string * 1000, greek_poly_string * 1000000]:
    modern_greek_str =  polytonic2modern_greek(inp_str)
    print(f"\nString size: {len(inp_str)}")
    %timeit modern_greek_str =  polytonic2modern_greek(inp_str)
    modern_greek_str =  polytonic2modern_greek(inp_str)
    %timeit modern_greek_str =  mes2ascii(inp_str)
```
Output:
```console
Creating autoaligned llemmatizers O(|src_alphabet|x|dst_alphabet|)
Medium llemmatizer: |34|x|186|
1.97 s ± 18.4 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
Large llemmatizer: |100|x|3549|
46.2 s ± 1 s per loop (mean ± std. dev. of 7 runs, 1 loop each)
    
Applying the medium and large llemmatizers on strings:

String size: 80
6.06 μs ± 48.1 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
5.94 μs ± 65 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)

String size: 80000
361 μs ± 6.79 μs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
397 μs ± 3.3 μs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)

String size: 80000000
499 ms ± 984 μs per loop (mean ± std. dev. of 7 runs, 1 loop each)
521 ms ± 13.2 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
```



## Command Line Invocation

### Evaluate Merges

```sh
ll_evaluate_merges -h # get help string with the cli interface
ll_evaluate_merges -corpus_glob  './sample_data/wienocist_charter_1/wienocist_charter_1*'
```

Attention the merge CER is not symetric at all!
```bash
# The following gives a CER of 0.0591
ll_evaluate_merges -corpus_glob  './sample_data/wienocist_charter_1/wienocist_charter_1*' -merges '[("I", "J"), ("i", "j")]'
# While the following gives a CER of 0.0007
ll_evaluate_merges -corpus_glob  './sample_data/wienocist_charter_1/wienocist_charter_1*' -merges '[("J", "I"), ("j", "i")]'
```

### Extract corpus alphabet
```bash
ll_extract_corpus_alphabet -h # get help string with the cli interface
ll_extract_corpus_alphabet -corpus_glob './sample_data/wienocist_charter_1/wienocist_charter_1*'
```

### Test corpus on alphabets
```bash
ll_test_corpus_on_alphabets -h # get help string with the cli interface
ll_test_corpus_on_alphabets -corpus_glob './sample_data/wienocist_charter_1/wienocist_charter_1*' -alphabets 'bmp_mufi,ascii,mes1,iso8859_2' -verbose
```

### Demapping

#### Setup
```bash
mkdir -p tmp/models
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -O ./tmp/tinyshakespeare.txt
cat ./tmp/tinyshakespeare.txt |shuf  --random-source ./tmp/tinyshakespeare.txt > ./tmp/tinyshakespeare_shuf.txt
head -n 1000 ./tmp/tinyshakespeare_shuf.txt > ./tmp/tinyshakespeare_test.txt
tail -n +1001 ./tmp/tinyshakespeare_shuf.txt > ./tmp/tinyshakespeare_exper.txt
```

#### Train a demapper
GPU is automatically employed if found
```bash
ll_train_one2one -corpus_files ./tmp/tinyshakespeare_exper.txt -output_model_path ./tmp/models/toy_model.pt -input_alphabet '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!",.:;? ' -output_alphabet '0abcdfghjklmnpqrstvwxz.' -nb_epochs 3
```
If a model has not been trained until nb_epochs, the training resumes.
```bash
ll_train_one2one -corpus_files ./tmp/tinyshakespeare_exper.txt -output_model_path ./tmp/models/toy_model.pt -input_alphabet '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!",.:;? ' -output_alphabet '0abcdfghjklmnpqrstvwxz.' -nb_epochs 50
```

#### Use a demapper
O demmaper can be use on streams or files
```bash
echo 'a da nat knaw what ta saa. bat a knaw what ta thank.' |ll_infer_one2one -model_path ./tmp/models/toy_model.pt
```


## Development

### Development Installation
For extending pylelemmatize, install from github.
```bash
git clone git@github.com:anguelos/pylelemmatize.git
cd pylelemmatize
pip install -r requirements
pip install -r ./docs/requirements.txt
pip install -e .
```
This will install pylelemmatize on your system in development mode.

### Testing
#### Running the unit tests
```bash
pytest --cov ./src/pylelemmatize/ ./test/pytest/
```

#### Running shell script tests

This will run all bash scripts with -h essetially checking syntax and imports
```bash
./test/test_shell_scripts.sh
```

