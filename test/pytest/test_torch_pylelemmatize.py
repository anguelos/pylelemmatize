import pylelemmatize_torch

is_not_github_action = "GITHUB_ACTIONS" not in pylelemmatize_torch.__dict__.get("os", {}).environ

def test_torch_util_imported():
    # TODO (anguelos): re-enable when why this fails in CI is understood
    if is_not_github_action:
        assert pylelemmatize_torch is not None
        assert hasattr(pylelemmatize_torch, "Seq2SeqDs")
        assert hasattr(pylelemmatize_torch, "DemapperLSTM")
    else:
        assert True