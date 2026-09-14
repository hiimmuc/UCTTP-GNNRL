"""`python -m horarium.cli.doctor`: what torch can see, and whether CUDA actually works."""

from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys

import torch

__all__ = ["main"]


def main(argv: list[str] | None = None) -> int:
    """Print the torch/CUDA/driver versions and check they line up.

    A non-zero exit when an NVIDIA driver is present but torch cannot use it is what a
    CUDA-version mismatch looks like: the wheel installs fine and then silently runs on the CPU.

    Args:
        argv: Command-line arguments, or `None` to use `sys.argv`.

    Returns:
        Exit code: 0 if CUDA works or is genuinely absent (unless `--require-cuda`), else 1.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-cuda", action="store_true", help="also fail when no GPU is present at all"
    )
    args = parser.parse_args(argv)

    driver = _driver_version()
    print(f"python          {platform.python_version()}")
    print(f"torch           {torch.__version__}")
    print(f"torch CUDA      {torch.version.cuda or 'cpu-only build'}")
    print(f"nvidia driver   {driver or 'not found'}")
    print(f"cuda.available  {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        major, minor = torch.cuda.get_device_capability(0)
        probe = torch.randn(256, 256, device="cuda")
        _ = (probe @ probe).sum().item()
        print(f"device          {name} (sm_{major}{minor})")
        print("compute check   ok")
        return 0

    if driver:
        print(
            f"\nA driver is installed but torch cannot use it. This torch is built for CUDA "
            f"{torch.version.cuda}; install the wheel matching the driver. This project pins "
            f"PyTorch's cu128 index in pyproject.toml for exactly this reason.",
            file=sys.stderr,
        )
        return 1
    if args.require_cuda:
        print("\nNo NVIDIA driver found and --require-cuda was given.", file=sys.stderr)
        return 1
    print("\nNo GPU found; everything will run on the CPU.")
    return 0


def _driver_version() -> str | None:
    """Driver version from nvidia-smi, or None when it is not installed or fails.

    Returns:
        The driver version string, or `None`.
    """
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return None
    completed = subprocess.run(  # noqa: S603
        [executable, "--query-gpu=driver_version", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0 or not completed.stdout.strip():
        return None
    return completed.stdout.strip().splitlines()[0]


if __name__ == "__main__":
    raise SystemExit(main())
