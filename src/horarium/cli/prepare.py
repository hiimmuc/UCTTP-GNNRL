"""`python -m horarium.cli.prepare`: download the corpus, verify it, parse it, write the docs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from horarium.data.download import download_instances, manifest_path, verify_instances
from horarium.data.ectt import read_ectt
from horarium.eval.stats import corpus_csv, corpus_markdown, summarise_corpus
from horarium.graph.features import feature_dictionary_markdown
from horarium.problem.checks import check_instance

__all__ = ["main"]


def main(argv: list[str] | None = None) -> int:
    """Fetch and validate the corpus, then regenerate the docs derived from it.

    Deterministic end to end, so it takes no --seed.

    Args:
        argv: Command-line arguments, or `None` to use `sys.argv`.

    Returns:
        Exit code: non-zero if any instance fails its structural checks.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--docs-dir", type=Path, default=Path("../docs"))
    parser.add_argument(
        "--force-download", action="store_true", help="re-fetch even if the manifest verifies"
    )
    args = parser.parse_args(argv)

    if args.force_download or not manifest_path(args.raw_dir).exists():
        print(f"downloading instances into {args.raw_dir} ...")
        download_instances(args.raw_dir)
    verify_instances(args.raw_dir)
    paths = sorted(args.raw_dir.rglob("*.ectt"))
    print(f"manifest verified: {len(paths)} instances")

    failed = 0
    warnings = 0
    for path in paths:
        result = check_instance(read_ectt(path))
        warnings += len(result.warnings)
        if not result.ok:
            failed += 1
            print(f"CHECK FAILED {path}:", file=sys.stderr)
            for problem in result.errors:
                print(f"  - {problem}", file=sys.stderr)
    print(f"checks: {len(paths) - failed}/{len(paths)} clean, {warnings} warnings")
    if failed:
        return 1

    args.docs_dir.mkdir(parents=True, exist_ok=True)
    summaries = summarise_corpus(args.raw_dir)
    (args.docs_dir / "feature_dictionary.md").write_text(feature_dictionary_markdown())
    (args.docs_dir / "instance_statistics.md").write_text(corpus_markdown(summaries))
    (args.docs_dir / "instance_statistics.csv").write_text(corpus_csv(summaries))
    print(f"wrote feature_dictionary.md and instance_statistics.{{md,csv}} to {args.docs_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
