# Copyright (C) 2024 Roberto Rossini <roberroso@uio.no>
#
# SPDX-License-Identifier: MIT

import pathlib
import warnings

import h5py
import hictkpy
import pytest

from stripepy import main

from .common import matplotlib_avail

testdir = pathlib.Path(__file__).resolve().parent.parent


@pytest.mark.end2end
class TestStripePyCall:
    @staticmethod
    def setup_class():
        test_files = [
            testdir / "data" / "4DNFI9GMP2J8.mcool",
        ]

        for f in test_files:
            if not f.exists():
                raise RuntimeError(
                    f'unable to find file "{f}". Did you download the test files prior to running pytest?'
                )

    @staticmethod
    def test_stripepy_call(tmpdir):
        tmpdir = pathlib.Path(tmpdir)
        testfile = testdir / "data" / "4DNFI9GMP2J8.mcool"
        resolution = 10_000

        chrom_sizes = hictkpy.MultiResFile(testfile).chromosomes()
        chrom_size_cutoff = sum(chrom_sizes.values()) // len(chrom_sizes)

        output_file = tmpdir / f"{testfile.stem}.hdf5"
        log_file = tmpdir / f"{testfile.stem}.log"

        args = [
            "call",
            str(testfile),
            str(resolution),
            "--glob-pers-min",
            "0.10",
            "--loc-pers-min",
            "0.33",
            "--loc-trend-min",
            "0.25",
            "--output-file",
            str(output_file),
            "--log-file",
            str(log_file),
            "--min-chrom-size",
            str(chrom_size_cutoff),
        ]
        main(args)

        assert output_file.is_file()
        assert h5py.File(output_file).attrs.get("format", "unknown") == "HDF5::StripePy"

        assert log_file.is_file()

    @staticmethod
    def test_stripepy_call_with_roi(tmpdir):
        tmpdir = pathlib.Path(tmpdir)
        testfile = testdir / "data" / "4DNFI9GMP2J8.mcool"
        resolution = 10_000

        chrom_size_cutoff = max(hictkpy.MultiResFile(testfile).chromosomes().values()) - 1

        output_file = tmpdir / f"{testfile.stem}.hdf5"
        log_file = tmpdir / f"{testfile.stem}.log"
        plot_dir = tmpdir / "plots"

        args = [
            "call",
            str(testfile),
            str(resolution),
            "--glob-pers-min",
            "0.10",
            "--loc-pers-min",
            "0.33",
            "--loc-trend-min",
            "0.25",
            "--output-file",
            str(output_file),
            "--log-file",
            str(log_file),
            "--plot-dir",
            str(plot_dir),
            "--min-chrom-size",
            str(chrom_size_cutoff),
            "--roi",
            "middle",
        ]
        if not matplotlib_avail():
            with pytest.raises(ImportError):
                main(args)
            pytest.skip("matplotlib not available")

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            main(args)

        assert output_file.is_file()
        assert h5py.File(output_file).attrs.get("format", "unknown") == "HDF5::StripePy"

        assert log_file.is_file()
        assert plot_dir.is_dir()
