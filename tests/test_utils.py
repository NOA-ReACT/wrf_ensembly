import pytest

from wrf_ensembly import utils
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


@pytest.mark.parametrize(
    "value, seconds",
    [
        ("30", 30 * 60),
        ("30:15", 30 * 60 + 15),
        ("12:00:00", 12 * 3600),
        ("01:02:03", 3723),
        ("1-0", 24 * 3600),
        ("2-03", 51 * 3600),
        ("1-02:30", 26 * 3600 + 30 * 60),
        ("1-02:30:05", 26 * 3600 + 30 * 60 + 5),
    ],
)
def test_parse_slurm_time(value, seconds):
    assert utils.parse_slurm_time(value) == seconds


@pytest.mark.parametrize("value", ["", "12h", "1-", "1:2:3:4", "-5"])
def test_parse_slurm_time_rejects_garbage(value):
    with pytest.raises(ValueError):
        utils.parse_slurm_time(value)


def test_format_slurm_time_rounds_up_and_parses_back():
    assert utils.format_slurm_time(3600) == "0-01:00:00"
    assert utils.format_slurm_time(3601) == "0-01:01:00"
    assert utils.format_slurm_time(26.5 * 3600) == "1-02:30:00"
    assert utils.parse_slurm_time(utils.format_slurm_time(7322)) == 7380
