"""End-to-end tests: the whole audit on synthetic data.

These run the real pipeline -- load, split, fit, measure, attribute, price, recalibrate, report --
on a small generated file, so a refactor that breaks the wiring fails here rather than on the
Kaggle data an hour later. They also check the two claims the report makes about recalibration
(monotone, so AUROC is unchanged; and the paired money interval is a paired one), because those
are properties of the pipeline, not of any single function.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from fraud_audit.cli import build_parser, main
from fraud_audit.config import AuditConfig
from fraud_audit.evaluate import EVAL_SPLITS
from fraud_audit.models import MODEL_NAMES
from fraud_audit.pipeline import run_audit

EXPECTED_TABLES = {
    "splits",
    "exclusions",
    "integrity",
    "duplicates",
    "discrimination",
    "auroc_change",
    "calibration",
    "drift_psi",
    "money_raw",
    "fixed_cutoff",
    "recalibration",
    "money_ci",
    "review_cost_sensitivity",
}


@pytest.fixture(scope="module")
def result(csv_path, tmp_path_factory):
    cfg = AuditConfig(
        data_path=csv_path,
        output_dir=str(tmp_path_factory.mktemp("audit")),
        n_boot=25,
        review_cost_grid=(1.0, 5.0),
        make_figures=False,
    )
    return run_audit(cfg)


class TestResultShape:
    def test_every_expected_table_is_produced(self, result):
        assert set(result.tables) == EXPECTED_TABLES

    def test_no_table_is_empty(self, result):
        empty = [name for name, frame in result.tables.items() if frame.empty]
        assert not empty

    def test_all_four_models_are_fitted_and_scored(self, result):
        assert set(result.predictions) == set(MODEL_NAMES)
        for name in MODEL_NAMES:
            for split in EVAL_SPLITS:
                probs = np.asarray(result.predictions[name][split])
                assert probs.shape == result.split.y[split].shape
                assert ((probs >= 0) & (probs <= 1)).all()

    def test_a_verdict_was_reached_on_both_dimensions(self, result):
        assert result.verdicts.ranking in {"HELD", "DEGRADED", "INCONCLUSIVE"}
        assert result.verdicts.calibration_oot in {"IN", "OUT", "UNCLEAR"}

    def test_the_duplicates_planted_in_the_fixture_were_removed(self, result):
        """The generator inserts 5 exact duplicate rows; the exclusions table must own up to them."""
        assert result.dataset.n_removed == 5


class TestReportIsGeneratedFromTheNumbers:
    def test_report_is_written_and_mentions_the_subject(self, result):
        assert result.config.subject in result.report
        assert "## 1. Verdict" in result.report

    def test_report_carries_the_data_fingerprint(self, result):
        assert result.dataset.sha256[:16] in result.report

    def test_no_placeholder_or_failed_interpolation_survives(self, result):
        for marker in ("{", "}", "nan%", "None"):
            assert marker not in result.report, marker

    def test_the_headline_auroc_in_the_prose_matches_the_table(self, result):
        subject = result.config.subject
        table = result.tables["discrimination"]
        oot = table[(table.model == subject) & (table.split == "oot")].iloc[0]
        assert f"{oot.auroc:.3f} out-of-time" in result.report

    def test_the_recommendation_is_not_empty(self, result):
        assert len(result.verdicts.recommendation) > 20
        assert result.verdicts.recommendation in result.report


class TestRecalibrationClaims:
    def test_the_primary_recalibration_leaves_ranking_untouched(self, result):
        """The report says so in section 4; it must be true, not hoped."""
        y = result.split.y["oot"]
        before = roc_auc_score(y, result.subject_raw_oot)
        after = roc_auc_score(y, result.subject_recalibrated_oot)
        assert after == pytest.approx(before, abs=1e-9)

    def test_every_model_got_every_recalibration_method(self, result):
        table = result.tables["recalibration"]
        assert len(table) == len(MODEL_NAMES) * 4  # three methods plus the raw baseline
        assert "(none: raw)" in set(table["method"])

    def test_the_paired_gain_interval_is_tighter_than_the_unpaired_difference(self, result):
        """Pairing on shared bootstrap indices is the point; an unpaired interval would be wider."""
        lo, hi = result.money["gain_ci"]
        raw_lo, raw_hi = result.money["savings_raw_ci"]
        assert (hi - lo) < (raw_hi - raw_lo)


class TestSaveArtefacts:
    def test_writes_report_tables_and_config(self, result):
        out = result.save()
        assert (out / "model_risk_audit_report.md").read_text(encoding="utf-8") == result.report
        for name in EXPECTED_TABLES:
            assert (out / "tables" / f"table_{name}.csv").exists()
        config = json.loads((out / "run_config.json").read_text(encoding="utf-8"))
        assert config["data_sha256"] == result.dataset.sha256
        assert config["verdicts"]["ranking"] == result.verdicts.ranking
        assert config["config"]["seed"] == result.config.seed

    def test_figures_are_rendered_when_asked(self, csv_path, tmp_path):
        cfg = AuditConfig(
            data_path=csv_path,
            output_dir=str(tmp_path / "figs"),
            n_boot=10,
            review_cost_grid=(5.0,),
            make_figures=True,
        )
        out = run_audit(cfg).save()
        rendered = sorted(p.name for p in (out / "figures").glob("*.png"))
        assert rendered == [
            "fig1_discrimination.png",
            "fig2_reliability.png",
            "fig3_recalibration.png",
            "fig4_money.png",
        ]
        assert all((out / "figures" / name).stat().st_size > 5_000 for name in rendered)


class TestDeterminism:
    def test_two_runs_with_the_same_seed_agree(self, csv_path, tmp_path):
        cfg = AuditConfig(
            data_path=csv_path,
            output_dir=str(tmp_path),
            n_boot=10,
            review_cost_grid=(5.0,),
            make_figures=False,
        )
        first, second = run_audit(cfg), run_audit(cfg)
        assert first.money["gain"] == pytest.approx(second.money["gain"])
        assert first.tables["discrimination"].equals(second.tables["discrimination"])


class TestCli:
    def test_parser_exposes_the_documented_flags(self):
        args = build_parser().parse_args(["--data", "x.csv", "--review-cost", "3", "--no-figures"])
        assert args.data_path == "x.csv"
        assert args.review_cost == 3.0
        assert args.make_figures is False

    def test_unset_flags_stay_none_so_the_yaml_wins(self):
        args = build_parser().parse_args([])
        assert args.seed is None and args.make_figures is None

    def test_runs_end_to_end_and_exits_zero(self, csv_path, tmp_path, capsys):
        code = main(["--data", csv_path, "--output", str(tmp_path / "cli"), "--n-boot", "10", "--no-figures"])
        assert code == 0
        assert "ranking:" in capsys.readouterr().out
        assert (tmp_path / "cli" / "model_risk_audit_report.md").exists()

    def test_a_missing_data_file_fails_cleanly(self, tmp_path, capsys):
        code = main(["--data", str(tmp_path / "nope.csv")])
        assert code == 2
        assert "does not exist" in capsys.readouterr().err
