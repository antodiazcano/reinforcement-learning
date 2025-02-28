"""
This script contains the test for the sum function.
"""

import pytest

from src.template_function import f


@pytest.mark.run(order=1)
@pytest.mark.parametrize("a, b", [(1, 2), (5, -1)])
def test_template(a: int, b: int) -> None:
    """
    Test for the sum function.
    """

    assert f(a, b) == a + b
