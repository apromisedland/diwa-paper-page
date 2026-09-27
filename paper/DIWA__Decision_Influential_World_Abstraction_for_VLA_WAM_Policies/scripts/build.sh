#!/usr/bin/env bash
set -euo pipefail
paper_root="$(cd "$(dirname "$0")/.." && pwd)"
if [[ -d "$paper_root/paper" ]]; then
  cd "$paper_root/paper"
else
  cd "$paper_root"
fi
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
mkdir -p "$paper_root/output/pdf"
cp main.pdf "$paper_root/output/pdf/DIWA_ICLR2027.pdf"
