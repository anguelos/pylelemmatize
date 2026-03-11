from typing import List, Tuple
import pytest
import torch
from pylelemmatize.demapper_lstm_ctc import ManyToMoreCollatorCTC
from pylelemmatize.many_to_more import ManyToMoreDS
#, align_sub_strings, banded_edit_path
import sys
import numpy as np


# Genomic alphabets are prefered for testing as they have small size and are well known
# Normally everything should extend to any alphabet. TODO (anguelos): add tests for larger alphabets

debug_many_to_more_collator_ctc = False

@pytest.mark.parametrize("textlines, max_unalignement, prefer_replication, ctc_epsilon_label, batch_inputs, batch_outputs", [
        [[("ACGT", "ACGT"), ("ACGT", "AGT"), ("ACGT", "ACGAGAGT"), ], -1, False, 0, # -1 in this case means 1, 1, and 5 
            [[[1, 0, 2, 0, 3, 0, 4, 0]], # [all src [src Batch [src Time Steps]]] 
             [[1, 0, 2, 0, 3, 0, 4, 0]], # [src Batch [src Time Steps]]
             [[1] + [0] * 5 + [2] + [0] * 5 + [3] + [0] * 5 + [4] + [0] * 5],],
            [[[1, 2, 3, 4]],
             [[1, 3, 4]],
             [[1, 2, 3, 1, 3, 1, 3, 4]],],
        ],

        [[("ACGT", "ACGT"), ("ACGT", "AGT"), ("ACGT", "ACGAGAGT"), ], -1, True, 0, # -1 in this case means 1, 1, and 5
            [[[1, 0, 2, 0, 3, 0, 4, 0]], 
             [[1, 0, 2, 0, 3, 0, 4, 0]],
             [[1]* 5 + [0] + [2] * 5 + [0] + [3] * 5 + [0] + [4] * 5 + [0]],],
            [[[1, 2, 3, 4]], 
             [[1, 3, 4]], 
             [[1, 2, 3, 1, 3, 1, 3, 4]],],
        ],

        [[("ACGT", "ACGT"), ("ACGT", "AGT"), ("ACGT", "ACGAGAGT"), ], 7, False, 0, # -1 in this case means 1, 1, and  
            [[[1] + [0] * 7 + [2] + [0] * 7 + [3] + [0] * 7 + [4] + [0] * 7], 
             [[1] + [0] * 7 + [2] + [0] * 7 + [3] + [0] * 7 + [4] + [0] * 7],
             [[1] + [0] * 7 + [2] + [0] * 7 + [3] + [0] * 7 + [4] + [0] * 7], ],
            [[[1, 2, 3, 4]],
             [[1, 3, 4]], 
             [[1, 2, 3, 1, 3, 1, 3, 4]],]
        ],

        [[("ACGT", "ACGT"), ("ACGT", "AGT"), ("ACGT", "ACGAGAGT"), ], 7, True, 0, # -1 in this case means 1, 1, and 6 
            [[[1] * 7 + [0] + [2] * 7 + [0] + [3] * 7 + [0] + [4] * 7 + [0]], 
             [[1] * 7 + [0] + [2] * 7 + [0] + [3] * 7 + [0] + [4] * 7 + [0]],
             [[1] * 7 + [0] + [2] * 7 + [0] + [3] * 7 +[0] + [4] * 7 + [0]], ],
            [[[1, 2, 3, 4]],
             [[1, 3, 4]],
             [[1, 2, 3, 1, 3, 1, 3, 4]],]
        ],
])
def test_many_to_more_collator_ctc(textlines, max_unalignement, prefer_replication, 
                                       ctc_epsilon_label, batch_inputs, batch_outputs):
    ds = ManyToMoreDS.create_from_aligned_textlines(line_pairs=textlines, min_src_len=3, min_tgt_len=3)
    assert len(ds) == len(textlines)  # Sanity check: All input output pairs became part of the dataset
    collator = ManyToMoreCollatorCTC(max_unalignment=max_unalignement, prefer_replication=prefer_replication, ctc_epsilon_label=ctc_epsilon_label)
    dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collator)
    for i, batch in enumerate(dl):
        srcs_tensor, tgts_tensor = batch
        if debug_many_to_more_collator_ctc:
            print("\nREAL SRC TENSOR:", repr(srcs_tensor), "\nTST SRC TENSOR:", repr(torch.tensor(batch_inputs[i], dtype=torch.long)))
            print("REAL TGT TENSOR:", repr(tgts_tensor), "\nTST TGT TENSOR:", repr(torch.tensor(batch_outputs[i], dtype=torch.long)))
        expected_srcs = torch.tensor(batch_inputs[i], dtype=torch.long)
        expected_tgts = torch.tensor(batch_outputs[i], dtype=torch.long)
        assert torch.equal(srcs_tensor, expected_srcs)
        assert torch.equal(tgts_tensor, expected_tgts)
