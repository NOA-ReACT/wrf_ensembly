import pytest

from wrf_ensembly.utils import int_to_letter_numeral


def test_int_to_letter_numeral():
    # Test case 1: i = 1
    assert int_to_letter_numeral(1) == "AAA"

    # Test case 2: i = 26
    assert int_to_letter_numeral(26) == "AAZ"

    # Test case 3: i = 27
    assert int_to_letter_numeral(27) == "ABA"

    # Test case 4: i = 28
    assert int_to_letter_numeral(28) == "ABB"

    # Test case 4: i = 53
    assert int_to_letter_numeral(53) == "ACA"

    # Test case 5: i = 676
    assert int_to_letter_numeral(676) == "AZZ"

    # Test case 6: i = 677
    assert int_to_letter_numeral(677) == "BAA"

    # Test case 7: i = 17576
    assert int_to_letter_numeral(17576) == "ZZZ"

    # Test case 8: i = 17577
    with pytest.raises(ValueError):
        int_to_letter_numeral(17577)
