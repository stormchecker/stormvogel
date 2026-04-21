"""Tests for stormvogel.model.action."""

from hypothesis import given, strategies as st

from stormvogel.model.action import Action, EmptyAction


def test_empty_action_label_is_none():
    a = Action(None)
    assert a.label is None


def test_action_empty_string_becomes_none():
    a = Action("")
    assert a.label is None


def test_str_empty_action():
    assert str(EmptyAction) == "Action(None)"


def test_str_nonempty_action():
    a = Action("go")
    assert str(a) == "Action('go')"


def test_repr():
    a = Action("hunt")
    assert repr(a) == "Action('hunt')"


def test_repr_empty():
    assert repr(EmptyAction) == "Action(None)"


def test_lt_none_label_is_less_than_real_label():
    a = Action(None)
    b = Action("z")
    assert a < b
    assert not b < a


def test_lt_compares_labels_lexicographically():
    a = Action("abc")
    b = Action("bcd")
    assert a < b
    assert not b < a


def test_lt_returns_not_implemented_for_non_action():
    a = Action("x")
    result = a.__lt__("not_an_action")
    assert result is NotImplemented


def test_equality_same_label():
    assert Action("x") == Action("x")


def test_inequality_different_labels():
    assert Action("x") != Action("y")


@given(st.text())
def test_action_str_roundtrip(label):
    """For non-empty labels, str(Action(label)) == label."""
    a = Action(label)
    if label == "":
        assert str(a) == "Action(None)"
    else:
        assert str(a) == f"Action({label!r})"


@given(st.text(min_size=1), st.text(min_size=1))
def test_action_lt_antisymmetry(l1, l2):
    """If a < b, then not b < a."""
    a = Action(l1)
    b = Action(l2)
    if a < b:
        assert not (b < a)
