# Print only the first 20 dimensions:

# python scripts/inspect_embeddings.py --max-values 20

# Or only 3 examples:

# python scripts/inspect_embeddings.py --examples 3

# For random patients rather than the first five:

# python scripts/inspect_embeddings.py --examples 5 --random-examples

# For a faster run without cosine similarity:

# python scripts/inspect_embeddings.py --skip-similarity

# python scripts/inspect_embeddings.py file1.npz file2.npz file3.npz file4.npz

# or:

# python scripts/inspect_embeddings.py --files file1.npz file2.npz file3.npz file4.npz

# You can even give a directory:

# python scripts/inspect_embeddings.py outputs/embeddings/

"""
Fed-THOR Embedding Inspection Utility
=====================================

Inspects saved time-series, tabular, and fused embeddings.

Input files:
    The inspector accepts any number of NPZ files. The original three
    Fed-THOR files remain the defaults when no files are supplied.
    You can pass files positionally, with --files, or supply a directory
    containing NPZ files.

The script reports:
    - NPZ contents
    - embedding shapes and dimensions
    - NaN / Inf checks
    - example embedding vectors
    - PCA explained variance
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_TAB = Path("outputs/embeddings/tabular_embeddings.npz")
DEFAULT_TAB_MLP = Path("outputs/embeddings/tabular_embeddings_mlp.npz")
DEFAULT_TS = Path("outputs/embeddings/timeseries_embeddings.npz")
DEFAULT_TS_RNN = Path("outputs/embeddings/timeseries_embeddings_rnn.npz")
DEFAULT_TS_TRANSFORMER = Path("outputs/embeddings/timeseries_embeddings_transformer.npz")
DEFAULT_TS_GRU = Path("outputs/embeddings/timeseries_embeddings_gru.npz")
DEFAULT_FUSED = Path("outputs/embeddings/fused_embeddings.npz")

DEFAULT_OUTPUT_DIR = Path("outputs/embedding_inspection")

RANDOM_SEED = 42


# =============================================================================
# Formatting helpers
# =============================================================================

def section(title: str) -> None:
    """Print a major section header."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# =============================================================================
# NPZ loading & detection
# =============================================================================

def load_npz(path: Path) -> np.lib.npyio.NpzFile:
    """Load an NPZ file and validate that it exists."""
    if not path.exists():
        raise FileNotFoundError(f"Embedding file not found:\n  {path.resolve()}")

    print(f"\nLoading: {path}")
    print(f"File size: {path.stat().st_size / (1024 ** 2):.2f} MB")

    return np.load(path, allow_pickle=False)


def detect_embedding_array(data: np.lib.npyio.NpzFile) -> tuple[str, np.ndarray]:
    """Detect the main 2-D numeric embedding array."""
    if "embeddings" in data.files:
        arr = data["embeddings"]
        if arr.ndim != 2:
            raise ValueError(f"'embeddings' exists but is not 2-D: {arr.shape}")
        return "embeddings", arr

    candidates = []
    for key in data.files:
        arr = data[key]
        if arr.ndim == 2 and np.issubdtype(arr.dtype, np.number):
            candidates.append((key, arr))

    if not candidates:
        raise ValueError("Could not find a 2-D numeric embedding array.")

    candidates.sort(key=lambda item: item[1].size, reverse=True)
    return candidates[0]


# =============================================================================
# 1. NPZ contents & embedding shapes/dimensions
# =============================================================================

def print_npz_contents(name: str, data: np.lib.npyio.NpzFile) -> None:
    """Print all keys, metadata, shapes, and dimensions."""
    section(f"{name.upper()} - NPZ CONTENTS")

    print(f"Keys found: {len(data.files)}")

    for key in data.files:
        arr = data[key]
        print(f"\n  Key: {key}")
        print(f"    shape : {arr.shape}")
        print(f"    dtype : {arr.dtype}")
        print(f"    ndim  : {arr.ndim}")
        print(f"    size  : {arr.size}")

    key, embeddings = detect_embedding_array(data)
    n_samples, dimension = embeddings.shape

    print("\nDetected embedding array:")
    print(f"  key                 = {key}")
    print(f"  shape               = {embeddings.shape}")
    print(f"  Number of samples   = {n_samples:,}")
    print(f"  Embedding dimension = {dimension}")


# =============================================================================
# 2. NaN / Inf checks
# =============================================================================

def check_nan_inf(name: str, embeddings: np.ndarray) -> None:
    """Perform NaN and Inf checks on embeddings."""
    section(f"{name.upper()} - NAN / INF CHECKS")

    nan_count = np.isnan(embeddings).sum()
    inf_count = np.isinf(embeddings).sum()

    print(f"Total values : {embeddings.size:,}")
    print(f"NaN values   : {nan_count:,}")
    print(f"Inf values   : {inf_count:,}")

    if nan_count == 0 and inf_count == 0:
        print("\n[OK] No NaN or Inf values detected.")
    else:
        print("\n[WARNING] Found invalid values (NaN or Inf) in the embeddings!")


# =============================================================================
# 3. Example embedding vectors
# =============================================================================

def print_example_embeddings(
    name: str,
    data: np.lib.npyio.NpzFile,
    num_examples: int = 5,
    max_values: int | None = None,
    random_examples: bool = False,
) -> None:
    """Print example embedding vectors."""
    _, embeddings = detect_embedding_array(data)

    patient_ids = data["patient_ids"] if "patient_ids" in data.files else None

    n_samples = len(embeddings)
    num_examples = min(num_examples, n_samples)

    if random_examples:
        rng = np.random.default_rng(RANDOM_SEED)
        indices = rng.choice(n_samples, size=num_examples, replace=False)
    else:
        indices = np.arange(num_examples)

    section(f"{name.upper()} - EXAMPLE EMBEDDINGS")

    print(f"Printing {num_examples} example embedding(s)")

    if max_values is None:
        print("Vector display: COMPLETE VECTOR")
    else:
        print(f"Vector display: FIRST {max_values} DIMENSIONS")

    for index in indices:
        print("\n" + "-" * 80)

        if patient_ids is not None:
            print(f"Patient ID : {patient_ids[index]}")

        print(f"Row index  : {index}")
        print(f"Dimension  : {embeddings[index].shape[0]}")

        vector = embeddings[index]

        if max_values is None:
            print("\nEmbedding vector:")
            print(vector)
        else:
            shown = vector[:max_values]
            print(f"\nFirst {len(shown)} values:")
            print(shown)
            if len(vector) > max_values:
                print(f"... ({len(vector) - max_values} more values)")


# =============================================================================
# 4. PCA explained variance
# =============================================================================

def run_pca(
    name: str,
    embeddings: np.ndarray,
    sample_size: int = 10000,
    components: int = 20,
) -> None:
    """Run PCA and print explained variance."""
    section(f"{name.upper()} - PCA EXPLAINED VARIANCE")

    n = len(embeddings)
    sample_size = min(sample_size, n)

    rng = np.random.default_rng(RANDOM_SEED)
    indices = rng.choice(n, size=sample_size, replace=False)
    sample = embeddings[indices]

    print(f"Using {sample_size:,} samples for PCA.")

    max_components = min(components, sample.shape[1], sample.shape[0])

    pca = PCA(n_components=max_components, random_state=RANDOM_SEED)
    pca.fit(sample)

    explained = pca.explained_variance_ratio_

    if len(explained) >= 1:
        print(f"PC1 explained variance  : {explained[0]:.6f}")

    if len(explained) >= 2:
        print(f"PC2 explained variance  : {explained[1]:.6f}")
        print(f"PC1 + PC2 total         : {np.sum(explained[:2]):.6f}")

    print("\nCumulative explained variance:")

    cumulative = np.cumsum(explained)

    for i, value in enumerate(cumulative, start=1):
        print(f"  {i:2d} components : {value:.6f}")


# =============================================================================
# Path helpers
# =============================================================================

def _expand_input_paths(paths: list[Path]) -> list[Path]:
    """Expand files/directories into a de-duplicated list of NPZ files."""
    expanded: list[Path] = []
    seen: set[Path] = set()

    for path in paths:
        path = path.expanduser()

        if path.is_dir():
            candidates = sorted(p for p in path.rglob("*.npz") if p.is_file())
        else:
            candidates = [path]

        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved not in seen:
                seen.add(resolved)
                expanded.append(candidate)

    if not expanded:
        raise FileNotFoundError("No NPZ embedding files were supplied/found.")

    return expanded


def _display_name(path: Path, index: int) -> str:
    """Create a readable and unique inspection name."""
    stem = path.stem.replace("_", " ").replace("-", " ").strip()
    return stem.title() if stem else f"Embedding {index}"


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fed-THOR embedding inspection utility. "
            "Inspect any number of NPZ embedding files."
        )
    )

    parser.add_argument(
        "files",
        nargs="*",
        type=Path,
        help="One or more NPZ files or directories containing NPZ files.",
    )

    parser.add_argument(
        "--files",
        dest="input_files",
        nargs="+",
        type=Path,
        default=None,
        help="One or more NPZ files/directories to inspect.",
    )

    # Legacy path arguments retained for backward compatibility
    parser.add_argument("--timeseries", type=Path, default=None, help="Path to time-series embeddings NPZ.")
    parser.add_argument("--tabular", type=Path, default=None, help="Path to tabular embeddings NPZ.")
    parser.add_argument("--fused", type=Path, default=None, help="Path to fused embeddings NPZ.")

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for inspection outputs.",
    )

    parser.add_argument(
        "--examples",
        type=int,
        default=5,
        help="Number of example embeddings to print per file. Default: 5.",
    )

    parser.add_argument(
        "--max-values",
        type=int,
        default=None,
        help="Maximum number of embedding values to print. Default: print complete vectors.",
    )

    parser.add_argument(
        "--random-examples",
        action="store_true",
        help="Print random examples instead of the first examples.",
    )

    parser.add_argument(
        "--pca-sample",
        type=int,
        default=10000,
        help="Number of embeddings used for PCA. Default: 10000.",
    )

    parser.add_argument(
        "--skip-pca",
        action="store_true",
        help="Skip PCA calculation.",
    )

    # Flags retained for CLI compatibility
    parser.add_argument("--skip-similarity", action="store_true", help="Retained CLI option (cosine similarity is removed).")
    parser.add_argument("--similarity-sample", type=int, default=2000, help="Retained CLI option.")
    parser.add_argument("--tsne", action="store_true", help="Retained CLI option.")
    parser.add_argument("--tsne-sample", type=int, default=2000, help="Retained CLI option.")
    parser.add_argument("--skip-duplicate-id-analysis", action="store_true", help="Retained CLI option.")

    args = parser.parse_args()

    # Input resolution
    supplied_files: list[Path] = []

    if args.input_files:
        supplied_files.extend(args.input_files)

    if args.files:
        supplied_files.extend(args.files)

    if not supplied_files:
        supplied_files = [
            path for path in (
                args.timeseries or DEFAULT_TS,
                args.tabular or DEFAULT_TAB,
                args.fused or DEFAULT_FUSED,
            )
        ]
    else:
        for path in (args.timeseries, args.tabular, args.fused):
            if path is not None:
                supplied_files.append(path)

    input_paths = _expand_input_paths(supplied_files)

    # -------------------------------------------------------------------------
    # Execution
    # -------------------------------------------------------------------------

    section("EMBEDDING INSPECTION")

    print("Fed-THOR / Multimodal Embedding Inspector")
    print(f"\nTotal input files: {len(input_paths)}")

    for i, path in enumerate(input_paths, start=1):
        print(f"  [{i}] {path}")

    datasets = []

    for index, path in enumerate(input_paths, start=1):
        name = _display_name(path, index)

        try:
            data = load_npz(path)
            key, embeddings = detect_embedding_array(data)

            # 1. NPZ contents & embedding shapes and dimensions
            print_npz_contents(name, data)

            # 2. NaN / Inf checks
            check_nan_inf(name, embeddings)

            # 3. Example embedding vectors
            print_example_embeddings(
                name,
                data,
                num_examples=args.examples,
                max_values=args.max_values,
                random_examples=args.random_examples,
            )

            # 4. PCA explained variance
            if not args.skip_pca:
                run_pca(
                    name,
                    embeddings,
                    sample_size=args.pca_sample,
                )

            datasets.append((name, path, embeddings))
            data.close()

        except Exception as exc:
            print(f"\n[ERROR] Could not inspect {path}: {exc}")

    # Final summary
    section("FINAL SUMMARY")
    print(f"Successfully inspected: {len(datasets)} / {len(input_paths)} files")

    for index, (name, path, embeddings) in enumerate(datasets, start=1):
        print(
            f"  [{index}] {name}: "
            f"{embeddings.shape[0]:,} samples × {embeddings.shape[1]} dimensions"
        )
        print(f"      {path}")

    print("\nInspection complete.")


if __name__ == "__main__":
    main()