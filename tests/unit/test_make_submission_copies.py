"""Tests for scripts/make_submission_copies.py (derived submission copies; sources are never touched)."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import make_submission_copies as m  # noqa: E402

RESULTS = "# Fed-THOR: results section (draft)\n\nintro\n\n## 1. Protocol\n\ntext about Table 3.\n\n## 7. Figures\n\nfig list\n\n## 8. Revision notes (delete before submission)\n\n- v1 note\n- v2 note\n"
ABSTRACT = ("# Fed-THOR: abstract and discussion (draft)\n\n*Draft status: delete this note and the bracketed pointers before submission.*\n\n## Abstract\n\nFirst sentence [Table 10, Section 3]. Second sentence [item 4] and more [Table 11a].\n\n"
            "## Discussion\n\nA claim (Table 6 shows a plateau) stays [Section 5]; another [Tables 11a-11c]: tail.\n")


class TestSubmissionCopies(unittest.TestCase):
    def test_results_section_8_removed_and_the_rest_kept(self):
        out = m.results_submission(RESULTS)
        self.assertNotIn("## 8.", out); self.assertNotIn("v2 note", out); self.assertNotIn("(draft)", out)
        for keep in ("## 1. Protocol", "text about Table 3.", "## 7. Figures", "fig list"):
            self.assertIn(keep, out)
        self.assertTrue(out.endswith("fig list\n"))

    def test_results_needs_exactly_one_section_8(self):
        with self.assertRaises(ValueError): m.results_submission("# t\n\n## 1. a\n")
        with self.assertRaises(ValueError): m.results_submission(RESULTS + "\n## 8. again\n")

    def test_abstract_pointers_and_draft_note_removed_without_leaving_stray_spaces(self):
        out = m.abstract_submission(ABSTRACT)
        self.assertNotIn("Draft status", out); self.assertNotIn("(draft)", out)
        self.assertNotRegex(out, r"\[(?:Table|Tables|Section|item)")
        self.assertIn("First sentence. Second sentence and more.", out)
        self.assertIn("A claim (Table 6 shows a plateau) stays; another: tail.", out)      # ordinary in-text references are kept
        self.assertNotIn(" .", out); self.assertNotIn("  ", out)

    def test_files_are_derived_and_sources_untouched(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d); (d / "r.md").write_text(RESULTS, encoding="utf-8"); (d / "a.md").write_text(ABSTRACT, encoding="utf-8")
            self.assertEqual(m.main(["--results", str(d / "r.md"), "--package-results", str(d / "r.md"), "--abstract", str(d / "a.md")]), 0)
            self.assertEqual((d / "r.md").read_text(encoding="utf-8"), RESULTS); self.assertEqual((d / "a.md").read_text(encoding="utf-8"), ABSTRACT)
            self.assertTrue((d / "r_submission.md").exists() and (d / "a_submission.md").exists())
            self.assertNotIn("## 8.", (d / "r_submission.md").read_text(encoding="utf-8"))

    def test_a_leftover_draft_marker_stops_the_run(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d); (d / "a.md").write_text(ABSTRACT + "\nStray line (draft) here.\n", encoding="utf-8")
            with self.assertRaises(ValueError): m.make(d / "a.md", "abstract")
            self.assertFalse((d / "a_submission.md").exists())                               # nothing is written when the check fails


if __name__ == "__main__":
    unittest.main()
