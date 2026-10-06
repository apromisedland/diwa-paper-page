import collections

import pytest

from utils.real_dataset import resolve_real_dataset_adapter


def test_real_dataset_adapter_requires_explicit_class():
    with pytest.raises(RuntimeError, match="real_dataset_adapter"):
        resolve_real_dataset_adapter(None)
    with pytest.raises(ValueError, match="module.path:ClassName"):
        resolve_real_dataset_adapter("invalid")


def test_real_dataset_adapter_resolves_class_and_rejects_function():
    assert resolve_real_dataset_adapter("collections:Counter") is collections.Counter
    with pytest.raises(TypeError, match="dataset class"):
        resolve_real_dataset_adapter("collections:namedtuple")
