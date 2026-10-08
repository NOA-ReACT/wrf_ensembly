import datetime as dt

import pytest

from wrf_ensembly import segments
from wrf_ensembly.cycling import CycleInformation

START = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)


def make_cycles(n: int, hours: int = 6, last_hours: int | None = None):
    cycles = []
    t = START
    for i in range(n):
        length = last_hours if (last_hours is not None and i == n - 1) else hours
        end = t + dt.timedelta(hours=length)
        cycles.append(
            CycleInformation(
                start=t,
                end=end,
                cycle_offset=t - START,
                index=i,
                output_interval=None,
                forecast_end=end,
            )
        )
        t = end
    return cycles


def plan(cycles, first=0, stops=None, rate=100.0, max_walltime=12 * 3600, **kwargs):
    if stops is None:
        stops = segments.find_stops(len(cycles), set(), set())
    return segments.plan_segment(
        cycles,
        first,
        stops,
        rate_s_per_sim_hour=rate,
        rate_source="test",
        max_walltime_s=max_walltime,
        safety_factor=kwargs.pop("safety_factor", 1.0),
        checkpoint_interval_hours=kwargs.pop("checkpoint_interval_hours", 24),
        **kwargs,
    )


def test_find_stops_reasons_and_last_cycle():
    stops = segments.find_stops(10, {2, 5}, {5, 7}, run_until=7)

    assert stops == {
        2: segments.STOP_OBSERVATIONS,
        5: segments.STOP_OBSERVATIONS,
        7: segments.STOP_KEEP_RESTART,
        9: segments.STOP_LAST_CYCLE,
    }


def test_find_stops_ignores_cycles_outside_the_experiment():
    assert segments.find_stops(3, {-1, 7}, {10}, run_until=4) == {
        2: segments.STOP_LAST_CYCLE
    }


def test_segment_runs_to_the_next_stop():
    cycles = make_cycles(20)
    stops = segments.find_stops(20, {8}, set())

    p = plan(cycles, first=3, stops=stops)

    assert (p.first, p.last, p.stop_reason) == (3, 8, segments.STOP_OBSERVATIONS)
    assert p.estimated_walltime_s == pytest.approx(6 * 6 * 100)


def test_segment_on_a_stop_is_one_cycle():
    cycles = make_cycles(20)
    stops = segments.find_stops(20, {3}, set())

    p = plan(cycles, first=3, stops=stops)

    assert (p.first, p.last) == (3, 3)


def test_segment_runs_to_the_last_cycle():
    cycles = make_cycles(5)

    p = plan(cycles, first=1)

    assert (p.last, p.stop_reason) == (4, segments.STOP_LAST_CYCLE)


def test_segment_limited_by_walltime():
    cycles = make_cycles(40)
    # 600 s per simulated hour, 6 h cycles: 1 h per cycle. 12 h fit 12 cycles, but
    # with a safety factor of 1.3 only 9 (9 x 1.3 = 11.7)
    p = plan(cycles, rate=600, safety_factor=1.3)

    assert (p.first, p.last, p.stop_reason) == (0, 8, segments.STOP_WALLTIME)


def test_segment_has_one_cycle_even_if_it_does_not_fit():
    cycles = make_cycles(5)

    p = plan(cycles, rate=10_000)

    assert (p.first, p.last, p.stop_reason) == (0, 0, segments.STOP_WALLTIME)


def test_run_until_is_a_stop():
    cycles = make_cycles(20)
    stops = segments.find_stops(20, set(), set(), run_until=6)

    p = plan(cycles, first=2, stops=stops, run_until=6)

    assert (p.last, p.stop_reason, p.run_until) == (6, segments.STOP_RUN_UNTIL, 6)


@pytest.mark.parametrize(
    "segment_min, max_min, expected",
    [
        (6 * 60, 24 * 60, 6 * 60),  # shorter than the interval: only at the end
        (48 * 60, 24 * 60, 24 * 60),
        (42 * 60, 24 * 60, 6 * 60),  # 7 cycles, prime: every cycle
        (30 * 60, 24 * 60, 6 * 60),  # 5 cycles, prime: every cycle
        (36 * 60, 24 * 60, 18 * 60),
    ],
)
def test_checkpoint_interval_on_cycle_boundaries(segment_min, max_min, expected):
    assert segments.checkpoint_interval(segment_min, max_min, 6 * 60) == expected


def test_checkpoint_interval_with_short_last_cycle():
    # 4 x 6 h + 3 h = 27 h: no multiple of 6 h divides it, 13.5 h does
    assert segments.checkpoint_interval(27 * 60, 24 * 60, 6 * 60) == 810
    # 24 h + 37 min = 1477 min = 7 x 211: 211 min
    assert segments.checkpoint_interval(1477, 24 * 60, 6 * 60) == 211
    # Prime number of minutes: no checkpoints before the end
    assert segments.checkpoint_interval(1499, 24 * 60, 6 * 60) == 1499


def test_plan_checkpoint_interval():
    cycles = make_cycles(20)

    p = plan(cycles, checkpoint_interval_hours=12)

    # 20 cycles x 6 h = 120 h fit with 100 s/h; 12 h divides 120 h
    assert p.last == 19
    assert p.checkpoint_interval_min == 12 * 60


def test_plan_checkpoint_interval_with_short_last_cycle():
    cycles = make_cycles(5, last_hours=3)

    p = plan(cycles)

    assert p.checkpoint_interval_min == 810  # 27 h / 2


def test_plan_roundtrip():
    cycles = make_cycles(10)
    stops = segments.find_stops(10, set(), set(), run_until=5)
    p = plan(cycles, first=2, stops=stops, run_until=5)

    assert segments.SegmentPlan.from_dict(p.to_dict()) == p
    assert 2 in p and 5 in p and 6 not in p and 1 not in p
    assert list(p.cycles) == [2, 3, 4, 5]
