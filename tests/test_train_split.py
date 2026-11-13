"""The train/test split must not leak one source IP across both sides.

A plain `train_test_split` scatters flows from the same source into train and
test, so the model can memorise the attacker and report an optimistic score.
`split_train_test` prefers a session-aware split, then temporal, then a
stratified random fallback.
"""

import pandas as pd

from preprocess import features as F
from training.train_rf import split_train_test


def _frame(*, with_src=True, with_time=False, sources=8, rows_per=5):
    rows = []
    for source in range(sources):
        label = "recon" if source % 2 else "normal"
        for row in range(rows_per):
            record = {column: float(row + 1) for column in F.FEATURE_COLUMNS}
            record["label"] = label
            if with_src:
                record["src"] = f"10.0.0.{source}"
            if with_time:
                record["recv_ts"] = f"2026-10-16T19:52:{source:02d}"
            rows.append(record)
    return pd.DataFrame(rows)


def _xy(frame):
    y = pd.Series(
        ["Reconnaissance" if value == "recon" else "Normal" for value in frame["label"]]
    )
    X = F.select_features(frame.drop(columns=["label"]))
    return X, y


def test_session_split_keeps_sources_on_one_side():
    frame = _frame(with_src=True)
    X, y = _xy(frame)

    X_train, X_test, _, _, strategy = split_train_test(frame, X, y, test_size=0.25, random_state=42)

    assert "session-aware" in strategy
    train_sources = set(frame.loc[X_train.index, "src"])
    test_sources = set(frame.loc[X_test.index, "src"])
    assert train_sources and test_sources
    assert train_sources.isdisjoint(test_sources)


def test_temporal_split_puts_the_latest_rows_in_test():
    frame = _frame(with_src=False, with_time=True)
    X, y = _xy(frame)

    X_train, X_test, _, _, strategy = split_train_test(frame, X, y, test_size=0.25, random_state=42)

    assert "temporal" in strategy
    assert frame.loc[X_train.index, "recv_ts"].max() <= frame.loc[X_test.index, "recv_ts"].min()


def test_falls_back_to_stratified_random_without_session_or_time():
    frame = _frame(with_src=False, with_time=False)
    X, y = _xy(frame)

    _, _, y_train, y_test, strategy = split_train_test(frame, X, y, random_state=42)

    assert "stratified random" in strategy
    assert y_train.nunique() >= 2 and y_test.nunique() >= 2


def test_split_is_deterministic_for_a_fixed_seed():
    frame = _frame(with_src=True)
    X, y = _xy(frame)

    first = split_train_test(frame, X, y, test_size=0.25, random_state=7)
    second = split_train_test(frame, X, y, test_size=0.25, random_state=7)

    assert list(first[0].index) == list(second[0].index)
    assert list(first[1].index) == list(second[1].index)
