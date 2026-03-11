import pytest
import pylelemmatize as LL


@pytest.mark.parametrize("word,levels,map_str", [
    ("AAAAA", [1,2,3,5], "ACGT"),
])
def test_lemmatize_word_avg(word, levels, map_str):
    mapper = LL.llemmatizer(map_str)
    histogram_count = sum(levels)
    histogram_size = len(mapper)
    
    avg_phoc = LL.PHOC(levels=levels, mapper=mapper, normalization_mode="avg")
    avg_embedding = avg_phoc(word)
    assert list(avg_embedding.size()) == [1, histogram_size * sum(levels)]  #  sum(levels)
    assert avg_embedding.sum() == histogram_count


@pytest.mark.parametrize("word,levels,map_str,mode,target", [
    ("AAAT", [1], "ACGT","avg", [0.0, 0.0, 0.75, 0.0, 0.0, 0.25]),
    ("AAAT", [1], "ACGT","sum", [0.0, 0.0, 3.0, 0.0, 0.0, 1.0]),
    ("AAAT", [1], "ACGT","bin", [0.0, 0.0, 1.0, 0.0, 0.0, 1.0]),
    ("AaAT", [1], "ACGT","avg", [.25, 0.0, 0.5, 0.0, 0.0, 0.25]),
])
def test_lemmatize_word_exact(word, levels, map_str, mode, target):
    mapper = LL.llemmatizer(map_str)
    phoc = LL.PHOC(levels=levels, mapper=mapper, normalization_mode=mode)
    embedding = phoc(word).numpy()[0, :].tolist()
    print(embedding)
    print(target)
    assert embedding == pytest.approx(target, rel=1e-10)
