# Getting Started

Welcome to PyLeLemmatize! This guide will help you get started with installing and using the library.

## Prerequisites

Before you begin, ensure you have the following:
- Python 3.7 or higher installed.
- `pip` package manager.

## Installation

To install PyLeLemmatize, run the following command:

```bash
pip install git+https://github.com/anguelos/pylelemmatize.git
```

## Basic Usage

Here’s a quick example to get you started:

```python
import pylelemmatize

lemmatizer = pylelemmatize.create_lemmatizer("ACGTacgt", "ACGT", unknown_chr='.')

seq1 = "AACCGGT"
seq2 = "aagcgCT"

print(sum([n==k for n,k in zip(lemmatizer(seq1), lemmatizer(seq2))))
```

## Support

If you encounter any issues, feel free to open an issue on our [GitHub repository](https://github.com/anguelos/pylelemmatize).

Happy hacking!