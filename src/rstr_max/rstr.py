from sys import stderr
from typing import Generator, Iterable, Set, Optional, Dict

from sklearn.feature_extraction.text import CountVectorizer



def half(vocab: list[tuple[str, int]], *, previous: Dict[str, int] = None) -> Optional[Dict[str, int]]:
    if not vocab:
        print("Warning: vocab is empty, half function will return `previous` !", file=stderr)
        return previous

    if previous is None:
        is_previous = False
        rstrs = {}
    else:
        assert isinstance(previous, dict), "previous should be a dict"
        is_previous = True
        rstrs = previous

    last_seen_word = None
    last_seen_count = None
    for word, count in vocab:
        if not last_seen_word:
            last_seen_word = word
            last_seen_count = count
            continue

        if last_seen_word in word:
            if last_seen_count == count:
                # Check if we have a previous list of rstrs
                # If we do, we check if the last_seen is in the previous list of rstrs
                # If it is, we remove it from the lisr as it is not a rstr
                # This happens because the first iteration is walking through the vocab list by lexographical order
                # whereas the second iteration is walking through the vocab list by reverse lexographical order
                # and a string could be extending from the right (first iteration) and from the left (second iteration)
                if is_previous:
                    pass
                    if last_seen_word in rstrs:
                        pass
                        del rstrs[last_seen_word]
                    last_seen_word = word
                    last_seen_count = count
                    continue

        rstrs[last_seen_word] = last_seen_count
        last_seen_word = None
        last_seen_count = None

    return rstrs


def max_repeated_substrings(
        corpora: str | list[str],
        *args,
        min_len: int = 1,
        min_count: int = 2,
        max_count: int = None,
        max_len: int = None,
        **kwargs
) -> list[tuple[str, int]]:
    # Argument validation
    if isinstance(corpora, list):
        if any(not isinstance(doc, str) for doc in corpora):
            raise ValueError("List should contain only strings")
        pass
    elif isinstance(corpora, str):
        corpora = [corpora]

    assert min_len > 0, "min_len should be equal or greater than 1"
    assert min_count > 0, "min_count should be equal or greater than 1"

    if max_len is None or max_count is None:
        max_ = max(map(len, corpora))
        max_len = max_len or max_
        max_count = max_count or max_
        del max_

    assert max_len >= min_len, "max_len should be equal or greater than min_len"
    assert max_count >= min_count, "max_count should be equal or greater than min_count"

    # Now we get to business
    cv = CountVectorizer(analyzer='char', ngram_range=(min_len, max_len))
    cv = cv.fit(corpora)

    counts = cv.transform(corpora).sum(axis=0).A1

    vocab = cv.get_feature_names_out()

    del cv

    vocab = sorted(zip(vocab, counts))

    result = half(vocab)

    vocab = sorted(vocab, reverse=True)

    result = half(vocab, previous=result)

    del vocab

    result = sorted(filter(lambda x: min_count <= x[1] <= max_count, result.items()), key=lambda x: x[1], reverse=True)

    return result

