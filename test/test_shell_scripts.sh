#!/bin/sh

# Simple script that simply tests if all the shell scripts can run.
# The actual output of each script is not tested here. The scripts are all launched
# with the -h option to ensure they can start properly.
# This testcase requires the package to be installed in the environment where it is run.
echo "Testing shell scripts... for syntax and imports"

ll_render_char_similarity_tree -h 2>/dev/null || (echo "ll_render_char_similarity_tree \033[0;31mFAILED\033[0m" && exit 1)
echo "ll_render_char_similarity_tree \033[0;32mOK\033[0m"

ll_infer_one2one -h 2>/dev/null || (echo "ll_infer_one2one \033[0;31mFAILED\033[0m" && exit 1)
echo "ll_infer_one2one \033[0;32mOK\033[0m"

ll_train_one2one -h 2>/dev/null || (echo "ll_train_one2one \033[0;31mFAILED\033[0m" && exit 1)
echo "ll_train_one2one \033[0;32mOK\033[0m"

ll_train_one2one_report -h 2>/dev/null || (echo "ll_train_one2one_report \033[0;31mFAILED\033[0m" && exit 1)
echo "ll_train_one2one_report \033[0;32mOK\033[0m"

ll_extract_corpus_alphabet -h 2>/dev/null || (echo "ll_extract_corpus_alphabet \033[0;31mFAILED\033[0m" && exit 1)
echo "ll_extract_corpus_alphabet \033[0;32mOK\033[0m"

#ll_test_corpus_on_alphabets -h 2>/dev/null || (echo "ll_test_corpus_on_alphabets \033[0;31mFAILED\033[0m" && exit 1)
#echo "ll_test_corpus_on_alphabets \033[0;32mOK\033[0m"

ll_evaluate_merges -h 2>/dev/null || (echo "ll_evaluate_merges \033[0;31mFAILED\033[0m" && exit 1)
echo "ll_evaluate_merges \033[0;32mOK\033[0m"

ll_extract_transcription_from_page_xml -h 2>/dev/null || (echo "ll_extract_transcription_from_page_xml \033[0;31mFAILED\033[0m" && exit 1)
echo "ll_extract_transcription_from_page_xml \033[0;32mOK\033[0m"

#ll_train_many_to_more -h 2>/dev/null || (echo "ll_train_many_to_more \033[0;31mFAILED\033[0m" && exit 1)
#echo "ll_train_many_to_more \033[0;32mOK\033[0m"

#ll_many_to_more -h 2>/dev/null || (echo "ll_many_to_more \033[0;31mFAILED\033[0m" && exit 1)
#echo "ll_many_to_more \033[0;32mOK\033[0m"

#ll_many_to_more_evaluate -h 2>/dev/null || (echo "ll_many_to_more_evaluate \033[0;31mFAILED\033[0m" && exit 1)
#echo "ll_many_to_more_evaluate \033[0;32mOK\033[0m"

ll_create_postcorrection_tsv -h 2>/dev/null || (echo "ll_create_postcorrection_tsv \033[0;31mFAILED\033[0m" && exit 1)
echo "ll_create_postcorrection_tsv \033[0;32mOK\033[0m"

ll_textline_full_cer -h 2>/dev/null || (echo "ll_textline_full_cer \033[0;31mFAILED\033[0m" && exit 1)
echo "ll_textline_full_cer \033[0;32mOK\033[0m"

ll_postcorrection -h 2>/dev/null || (echo "ll_postcorrection \033[0;31mFAILED\033[0m" && exit 1)
echo "ll_postcorrection \033[0;32mOK\033[0m"

ll_postcorrection_train -h 2>/dev/null || (echo "ll_postcorrection_train \033[0;31mFAILED\033[0m" && exit 1)
echo "ll_postcorrection_train \033[0;32mOK\033[0m"

