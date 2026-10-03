import struct
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from whale_coda_dynamics.rq3.experiments.report import (
    build_self_calibration_report,
    write_transfer_heatmap,
)


class ReportTests(unittest.TestCase):
    def test_calibration_report_includes_source_size(self):
        report = build_self_calibration_report(
            [{
                "source_dataset": "pacific", "clan": "FP", "selected": "true",
                "model_family": "unigram", "context_order": 0, "alpha": 0.1,
                "validation_coda_count": 4, "mean_bits_per_coda": -1,
                "mean_bits_per_symbol": -0.5,
            }],
            [{"source_dataset": "pacific", "clan": "FP"}] * 7,
        )
        self.assertIn("| pacific | FP | 7 |", report)

    def test_heatmap_is_png(self):
        rows = [
            {"source_clan": source, "target_clan": target,
             "mean_transfer_loss_bits_per_symbol": 0.0 if source == target else 1.0}
            for source in ("A", "B") for target in ("A", "B")
        ]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "heatmap.png"
            write_transfer_heatmap(path, rows)
            self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            self.assertEqual(struct.unpack(">I", path.read_bytes()[16:20])[0], 160)
