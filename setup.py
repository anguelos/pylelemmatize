from setuptools import setup, find_packages

version = open('src/pylelemmatize/version.py').read().split('=')[1].strip().strip('"')


setup(
    name='pylelemmatize',
    version=version,
    package_dir={"": "src"},           
    packages=find_packages(where="src"),
    install_requires=[
        'numpy', 'unidecode', 'fargv', 'matplotlib', 'scipy', 'tqdm', 'networkx', 'lxml'
    ],
    author='Anguelos Nicolaou',
    author_email='anguelos.nicolaou@gmail.com',
    description='A set utilities for hadling alphabets of corpora and OCR/HTR datasets',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/anguelos/pylelemmatize',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    license_files=["LICENSE*", "LICENSES/*"],
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
            'll_many_to_more=pylelemmatize.many_to_more:many_to_more_main',
            'll_many_to_more_evaluate=pylelemmatize.many_to_more:many_to_more_evaluate_main',
            
            'll_create_postcorrection_tsv=pylelemmatize.htr_postcorrection:main_create_postcorrection_tsv',
            
            'll_textline_full_cer=pylelemmatize.substitution_augmenter:main_textline_full_cer',
            'll_postcorrection=pylelemmatize.htr_postcorrection:main_postcorrection_infer',
            'll_postcorrection_train=pylelemmatize.htr_postcorrection:main_train_substitution_only_postcorrection',
        ],
    },
)