"""`python -m cinescope.pipeline` runs the classical stages without PyTorch installed."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pytest

from tests.test_week25 import synthetic_clip


class TestCliWithoutTorch(unittest.TestCase):
    def test_classical_stages_need_no_torch(self):
        pytest.importorskip("av")
        from cinescope.pipeline.__main__ import main

        with tempfile.TemporaryDirectory() as d:
            clip = Path(d) / "synth.mp4"
            synthetic_clip(clip)
            with mock.patch.dict(sys.modules, {"torch": None}):      # import torch -> ImportError
                rep = main([str(clip), "--stages", "shots", "palette", "--out", str(Path(d) / "out"), "--quiet"])
            self.assertEqual(len(rep["shots"]), 3)
            self.assertTrue((Path(d) / "out" / "synth" / "report.json").exists())


if __name__ == "__main__":
    unittest.main()
