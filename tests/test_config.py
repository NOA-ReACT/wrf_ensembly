from importlib import resources

import pytest

from wrf_ensembly import config


def make_config(per_member: dict) -> config.Config:
    template = resources.files("wrf_ensembly.config_templates").joinpath("default.toml")
    cfg = config.read_config(template, inject_environment=False)
    cfg.assimilation.n_members = 4
    cfg.wrf_namelist_per_member = per_member
    return cfg


@pytest.mark.parametrize("key", ["1", "member_1", "member_01", "member_001"])
def test_per_member_overrides_accept_index_and_dir_name(key):
    cfg = make_config({key: {"physics": {"mp_physics": 8}}})

    assert cfg.wrf_namelist_overrides_for_member(1) == {"physics": {"mp_physics": 8}}
    assert cfg.wrf_namelist_overrides_for_member(0) == {}


def test_per_member_overrides_merge_keys_for_same_member():
    cfg = make_config(
        {
            "2": {"physics": {"mp_physics": 8}},
            "member_02": {"physics": {"ra_lw_physics": 4}, "dynamics": {"diff_opt": 1}},
        }
    )

    assert cfg.wrf_namelist_overrides_for_member(2) == {
        "physics": {"mp_physics": 8, "ra_lw_physics": 4},
        "dynamics": {"diff_opt": 1},
    }


@pytest.mark.parametrize("key", ["member_04", "4", "mem_1", "member_x", "first"])
def test_per_member_overrides_reject_unknown_keys(key):
    cfg = make_config({key: {"physics": {"mp_physics": 8}}})

    with pytest.raises(ValueError, match=key):
        cfg.wrf_namelist_overrides_for_member(0)


def test_check_warns_about_cycling_t_instead_of_thm(caplog: pytest.LogCaptureFixture):
    cfg = make_config({})
    cfg.assimilation.cycled_variables = ["U", "V", "T", "QVAPOR"]

    cfg.check()

    assert "cycled_variables contains 'T' but not 'THM'" in caplog.text


@pytest.mark.parametrize(
    "template",
    [
        "default.toml",
        "iridium_chem_4.6.0.toml",
        pytest.param(
            "aris_chem_4.5.2.toml",
            marks=pytest.mark.xfail(reason="template has no [observations] section"),
        ),
    ],
)
def test_templates_pass_check(template: str, caplog: pytest.LogCaptureFixture):
    path = resources.files("wrf_ensembly.config_templates").joinpath(template)
    config.read_config(path, inject_environment=False)

    assert "THM" not in caplog.text


def test_check_requires_cycled_variables_in_wrfinput_mode():
    cfg = make_config({})
    cfg.assimilation.cycled_variables = []

    with pytest.raises(ValueError, match="cycled_variables is empty"):
        cfg.check()


def make_restart_config() -> config.Config:
    cfg = make_config({})
    cfg.assimilation.cycling_mode = "restart"
    cfg.assimilation.cycled_variables = []
    cfg.wrf_namelist.setdefault("physics", {})["sst_update"] = 1
    return cfg


def test_check_accepts_restart_mode(caplog: pytest.LogCaptureFixture):
    make_restart_config().check()

    assert caplog.text == ""


def test_check_rejects_restart_mode_without_hypsometric_opt_2():
    cfg = make_restart_config()
    cfg.wrf_namelist.setdefault("dynamics", {})["hypsometric_opt"] = 1

    with pytest.raises(ValueError, match="hypsometric_opt = 2"):
        cfg.check()


def test_check_warns_about_restart_mode_without_sst_update(
    caplog: pytest.LogCaptureFixture,
):
    cfg = make_restart_config()
    cfg.wrf_namelist["physics"]["sst_update"] = 0

    cfg.check()

    assert "without sst_update = 1" in caplog.text


def test_check_warns_about_cycled_variables_in_restart_mode(
    caplog: pytest.LogCaptureFixture,
):
    cfg = make_restart_config()
    cfg.assimilation.cycled_variables = ["THM"]

    cfg.check()

    assert "cycled_variables is ignored" in caplog.text
