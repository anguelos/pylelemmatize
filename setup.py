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
            'll_remap_alphabet=pylelemmatize:main_remap_alphabet',
            'll_render_char_similarity_tree=pylelemmatize:main_char_similarity_tree',
            'll_infer_one2one=pylelemmatize:main_infer_one2one',
            'll_train_one2one=pylelemmatize:main_train_one2one',
            'll_train_one2one_report=pylelemmatize:main_report_demapper',
            'll_extract_corpus_alphabet=pylelemmatize:main_alphabet_extract_corpus_alphabet',
            'll_test_corpus_on_alphabets=pylelemmatize:main_map_test_corpus_on_alphabets',
            'll_evaluate_merges=pylelemmatize:main_alphabet_evaluate_merges',
            'll_extract_transcription_from_page_xml=pylelemmatize.util:main_extract_transcription_from_page_xml',
        ],
    },
)