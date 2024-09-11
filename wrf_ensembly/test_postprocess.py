from pathlib import Path
import tempfile
from concurrent.futures import ThreadPoolExecutor

from wrf_ensembly.postprocess import apply_scripts_to_file


APPENDER_SCRIPT = """#!/usr/bin/env python
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--text", type=str, required=True)
parser.add_argument("input_file", type=str)
parser.add_argument("output_file", type=str)
args = parser.parse_args()

with open(args.input_file, "r") as f:
    text = f.read()

with open(args.output_file, "w") as f:
    f.write(text + args.text)
"""


def test_apply_scripts_to_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        appender_script = tmpdir / "appender.py"
        appender_script.write_text(APPENDER_SCRIPT)

        file_to_process = tmpdir / "file.nc"
        file_to_process.touch()

        workdir = tmpdir / "workdir"
        workdir.mkdir()

        scripts = [
            f"python {appender_script} --text=first {{in}} {{out}}",
            f"python {appender_script} --text=second {{in}} {{out}}",
            f"python {appender_script} --text=third {{in}} {{out}}",
        ]

        input_file, output_file = apply_scripts_to_file(
            scripts, file_to_process, workdir
        )

        print(input_file, output_file)

        assert input_file == file_to_process
        assert output_file.exists()
        assert output_file.name == "file.nc_script_2"
        assert output_file.read_text() == "firstsecondthird"
        assert (workdir / "file.nc_script_0").read_text() == "first"
        assert (workdir / "file.nc_script_1").read_text() == "firstsecond"
        assert (workdir / "file.nc_script_2").read_text() == "firstsecondthird"
        assert not (workdir / "file.nc_script_3").exists()


def test_apply_scripts_to_file_empty_scripts():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        file_to_process = tmpdir / "file.nc"
        file_to_process.touch()

        workdir = tmpdir / "workdir"

        scripts = []

        input_file, output_file = apply_scripts_to_file(
            scripts, file_to_process, workdir
        )

        assert input_file == file_to_process
        assert output_file.exists()
        assert output_file.name == "file.nc"
        assert output_file.read_text() == ""
        assert not (workdir / "file.nc_script_0").exists()


def test_apply_scripts_to_file_concurrent():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        appender_script = tmpdir / "appender.py"
        appender_script.write_text(APPENDER_SCRIPT)

        file_to_process_1 = tmpdir / "file1.nc"
        file_to_process_1.touch()

        file_to_process_2 = tmpdir / "file2.nc"
        file_to_process_2.touch()

        workdir_1 = tmpdir / "workdir1"
        workdir_1.mkdir()

        workdir_2 = tmpdir / "workdir2"
        workdir_2.mkdir()

        scripts = [
            f"python {appender_script} --text=first {{in}} {{out}}",
            f"python {appender_script} --text=second {{in}} {{out}}",
            f"python {appender_script} --text=third {{in}} {{out}}",
        ]

        def process_file(file_to_process, workdir):
            return apply_scripts_to_file(scripts, file_to_process, workdir)

        with ThreadPoolExecutor() as executor:
            future_1 = executor.submit(process_file, file_to_process_1, workdir_1)
            future_2 = executor.submit(process_file, file_to_process_2, workdir_2)

            input_file_1, output_file_1 = future_1.result()
            input_file_2, output_file_2 = future_2.result()

        assert input_file_1 == file_to_process_1
        assert output_file_1.exists()
        assert output_file_1.name == "file1.nc_script_2"
        assert output_file_1.read_text() == "firstsecondthird"
        assert (workdir_1 / "file1.nc_script_0").read_text() == "first"
        assert (workdir_1 / "file1.nc_script_1").read_text() == "firstsecond"
        assert (workdir_1 / "file1.nc_script_2").read_text() == "firstsecondthird"
        assert not (workdir_1 / "file1.nc_script_3").exists()

        assert input_file_2 == file_to_process_2
        assert output_file_2.exists()
        assert output_file_2.name == "file2.nc_script_2"
        assert output_file_2.read_text() == "firstsecondthird"
        assert (workdir_2 / "file2.nc_script_0").read_text() == "first"
        assert (workdir_2 / "file2.nc_script_1").read_text() == "firstsecond"
        assert (workdir_2 / "file2.nc_script_2").read_text() == "firstsecondthird"
        assert not (workdir_2 / "file2.nc_script_3").exists()
