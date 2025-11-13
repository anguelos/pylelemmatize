import pylelemmatize_torch

def test_torch_util_imported():
    print("KEYS:", dir(pylelemmatize_torch).keys())
    assert pylelemmatize_torch is not None
    assert hasattr(pylelemmatize_torch, "Seq2SeqDs")
    assert hasattr(pylelemmatize_torch, "DemapperLSTM")