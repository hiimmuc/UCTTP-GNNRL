"""Fetch the 61 public .ectt instances from a pinned upstream commit and hash them."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
import urllib.request
from pathlib import Path

REPOSITORY = "potassco/teaspoon"
COMMIT = "bf332c166b7019ea923559071928dc321e1559f4"
TARBALL_URL = f"https://codeload.github.com/{REPOSITORY}/tar.gz/{COMMIT}"

#: Every `bench/<family>_ectt/` directory upstream, and how many .ectt files it must contain.
EXPECTED_FAMILIES = {
    "DDS-2008": 7,
    "EasyAcademy": 12,
    "Erlangen": 6,
    "ITC-2007": 21,
    "Test": 5,
    "UUMCAS": 1,
    "UniUD": 9,
}
EXPECTED_TOTAL = 61


class DownloadError(RuntimeError):
    """The upstream archive did not contain what this module was pinned to expect."""


def manifest_path(raw_dir: Path) -> Path:
    """Location of the checksum manifest for a raw-data directory.

    Args:
        raw_dir: The raw-data directory (e.g. `data/raw`).

    Returns:
        The path `MANIFEST.json` is read from and written to.
    """
    return raw_dir.parent / "MANIFEST.json"


def sha256_of(path: Path) -> str:
    """Hex SHA-256 of a file's bytes.

    Args:
        path: File to hash.

    Returns:
        The lowercase hex digest.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download_instances(raw_dir: Path, *, url: str = TARBALL_URL) -> dict[str, str]:
    """Extract every bench/*.ectt file from the pinned archive into raw_dir/<family>/.

    Returns a mapping from raw_dir-relative path to SHA-256, which is also written to
    MANIFEST.json next to raw_dir.

    Args:
        raw_dir: Directory to extract instances into.
        url: HTTPS URL of the tarball to fetch.

    Returns:
        Mapping from raw_dir-relative path to SHA-256, matching MANIFEST.json.

    Raises:
        DownloadError: The archive is missing families or files relative to EXPECTED_FAMILIES.
    """
    if not url.startswith("https://"):
        msg = f"refusing to fetch over a non-HTTPS URL: {url}"
        raise DownloadError(msg)

    with urllib.request.urlopen(url) as response:  # noqa: S310 - scheme checked above
        archive = response.read()

    raw_dir.mkdir(parents=True, exist_ok=True)
    found: dict[str, int] = dict.fromkeys(EXPECTED_FAMILIES, 0)

    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        for member in tar.getmembers():
            family = _family_of(member.name)
            if family is None:
                continue
            if family not in EXPECTED_FAMILIES:
                msg = f"unexpected instance family {family!r} in {member.name}"
                raise DownloadError(msg)
            payload = tar.extractfile(member)
            if payload is None:
                msg = f"archive member {member.name} has no content"
                raise DownloadError(msg)
            target = raw_dir / family / Path(member.name).name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload.read())
            found[family] += 1

    if found != EXPECTED_FAMILIES:
        msg = f"instance counts changed upstream: expected {EXPECTED_FAMILIES}, got {found}"
        raise DownloadError(msg)

    manifest = {str(p.relative_to(raw_dir)): sha256_of(p) for p in sorted(raw_dir.rglob("*.ectt"))}
    if len(manifest) != EXPECTED_TOTAL:
        msg = f"expected {EXPECTED_TOTAL} instances on disk, found {len(manifest)}"
        raise DownloadError(msg)

    manifest_path(raw_dir).write_text(
        json.dumps(
            {"repository": REPOSITORY, "commit": COMMIT, "sha256": manifest},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return manifest


def verify_instances(raw_dir: Path) -> None:
    """Re-hash every file on disk against MANIFEST.json.

    Args:
        raw_dir: The raw-data directory to verify.

    Raises:
        DownloadError: A file is missing, extra, or its content changed.
    """
    path = manifest_path(raw_dir)
    if not path.exists():
        msg = f"no manifest at {path}; run the download first"
        raise DownloadError(msg)

    expected: dict[str, str] = json.loads(path.read_text())["sha256"]
    actual = {str(p.relative_to(raw_dir)): sha256_of(p) for p in sorted(raw_dir.rglob("*.ectt"))}

    if actual.keys() != expected.keys():
        missing = sorted(expected.keys() - actual.keys())
        extra = sorted(actual.keys() - expected.keys())
        msg = f"raw data does not match the manifest: missing={missing}, extra={extra}"
        raise DownloadError(msg)

    changed = sorted(name for name, digest in expected.items() if actual[name] != digest)
    if changed:
        msg = f"content changed since download: {changed}"
        raise DownloadError(msg)


def _family_of(member_name: str) -> str | None:
    """Family name for an archive path like `teaspoon-<sha>/bench/ITC-2007_ectt/comp01.ectt`.

    Args:
        member_name: Full path of a tar archive member.

    Returns:
        The family name (e.g. "ITC-2007"), or `None` if the path is not a `bench/*_ectt/*.ectt`
        entry.
    """
    parts = Path(member_name).parts
    if len(parts) < 3 or not member_name.endswith(".ectt"):  # noqa: PLR2004
        return None
    directory = parts[-2]
    if parts[-3] != "bench" or not directory.endswith("_ectt"):
        return None
    return directory.removesuffix("_ectt")
