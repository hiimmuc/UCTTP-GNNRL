#!/usr/bin/env bash
# The only shell script in this project. Everything else is `python -m horarium.cli.<command>`.
#
# Creates the virtualenv, installs the package with its dev + RL extras, and compiles the
# vendored solution validator. Safe to re-run; pass --rebuild to wipe .venv first.
#
#   scripts/setup.sh              # create/update .venv and build the validator
#   scripts/setup.sh --rebuild    # delete .venv and start clean (fixes mixed CUDA runtimes)
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_VERSION=3.12
VENV=.venv
VALIDATOR_SRC=external/validator/validator.cc
VALIDATOR=external/validator/validator

if [[ "${1:-}" == "--rebuild" ]]; then
	echo ">> removing $VENV"
	rm -rf "$VENV"
fi

if ! command -v uv >/dev/null 2>&1; then
	echo "error: uv is not installed -- see https://docs.astral.sh/uv/" >&2
	exit 1
fi

echo ">> creating virtualenv ($PYTHON_VERSION)"
uv venv --python "$PYTHON_VERSION" --allow-existing

echo ">> installing horarium[dev,rl,report]"
uv pip install --editable ".[dev,rl,report]"

if [[ ! -x "$VALIDATOR" || "$VALIDATOR_SRC" -nt "$VALIDATOR" ]]; then
	echo ">> compiling validator"
	if ! command -v g++ >/dev/null 2>&1; then
		echo "error: g++ is not installed; cannot build $VALIDATOR" >&2
		exit 1
	fi
	g++ -O2 -o "$VALIDATOR" "$VALIDATOR_SRC"
else
	echo ">> validator already current"
fi

echo
echo "done. Activate with:  source $VENV/bin/activate"
echo "then, for example:    python -m horarium.cli.check"
