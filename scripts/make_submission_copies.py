"""Make submission copies of the results text and the abstract/discussion draft. Sources are never modified.

  python scripts/make_submission_copies.py

Writes next to each source a copy named <name>_submission.md:
  * Fed-THOR_results_section.md and Fed-THOR_results_package/results_section.md : section 8 (revision notes) removed; " (draft)" dropped from the title.
  * Fed-THOR_abstract_and_discussion.md : the draft-status note removed; the bracketed pointers such as [Table 12] or [Section 5] removed; " (draft)" dropped from the title.
It stops with an error if a draft marker is still present in a copy afterwards. Re-run it after any change to the sources; the copies are derived, never edit them by hand.
"""
import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
POINTER = re.compile(r"[ \t]?\[(?:Tables?|Sections?|items?)\b[^\]\n]*\]")
LEFTOVER = [re.compile(p) for p in (r"^## 8\.", r"\[(?:Tables?|Sections?|items?)\b", r"^\*Draft status", r"\(draft\)", r"delete before submission")]


def strip_title_draft(text: str) -> str:
    lines = text.split("\n")
    for i, l in enumerate(lines):
        if l.startswith("# "):
            lines[i] = l.replace(" (draft)", "")
            break
    return "\n".join(lines)


def results_submission(text: str) -> str:
    lines = text.split("\n")
    cut = [i for i, l in enumerate(lines) if re.match(r"^## 8\.", l)]
    if len(cut) != 1:
        raise ValueError(f"expected exactly one '## 8.' heading, found {len(cut)}")
    return strip_title_draft("\n".join(lines[:cut[0]]).rstrip("\n") + "\n")


def abstract_submission(text: str) -> str:
    kept = [l for l in text.split("\n") if not l.startswith("*Draft status")]
    out = POINTER.sub("", "\n".join(kept))
    out = re.sub(r"\n{3,}", "\n\n", out)
    return strip_title_draft(out).rstrip("\n") + "\n"


def leftovers(text: str) -> list:
    return [(p.pattern, i + 1) for i, l in enumerate(text.split("\n")) for p in LEFTOVER if p.search(l)]


def make(src: Path, kind: str) -> Path:
    dst = src.with_name(src.stem + "_submission.md")
    if dst.resolve() == src.resolve():
        raise ValueError("refusing to overwrite a source file")
    out = (results_submission if kind == "results" else abstract_submission)(src.read_text(encoding="utf-8"))
    bad = leftovers(out)
    if bad:
        raise ValueError(f"{dst.name}: draft markers remain after cleaning: {bad[:5]}")
    dst.write_text(out, encoding="utf-8")
    return dst


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", default=str(REPO / "Fed-THOR_results_section.md"))
    ap.add_argument("--package-results", default=str(REPO / "Fed-THOR_results_package" / "results_section.md"))
    ap.add_argument("--abstract", default=str(REPO / "Fed-THOR_abstract_and_discussion.md"))
    a = ap.parse_args(argv)
    for path, kind in ((a.results, "results"), (a.package_results, "results"), (a.abstract, "abstract")):
        src = Path(path)
        before = src.read_text(encoding="utf-8")
        dst = make(src, kind)
        after = dst.read_text(encoding="utf-8")
        print(f"{src.name} -> {dst}: {len(before.splitlines())} -> {len(after.splitlines())} lines, {len(before) - len(after):,} characters removed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
