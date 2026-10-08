from pathlib import Path

from wrf_ensembly import jobfiles

from test_segments_experiment import make_experiment


def test_preprocess_jobfile_per_member_runs_chem_once(tmp_path: Path):
    exp = make_experiment(tmp_path / "exp", extra="")
    exp.cfg.data.per_member_meteorology = True

    lines = jobfiles.generate_preprocess_jobfile(exp).read_text().splitlines()

    chem = [line for line in lines if "interpolate-chem" in line]
    assert len(chem) == 1 and "--member" not in chem[0]
    assert lines.index(chem[0]) > lines.index("done")
    assert any("preprocess real --member $MEMBER" in line for line in lines)
