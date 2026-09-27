"""Package the anonymous manuscript with the inputs needed to regenerate results."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper" if (ROOT / "paper").is_dir() else ROOT
target = ROOT / "output/DIWA_ICLR2027_source.zip"
target.parent.mkdir(parents=True, exist_ok=True)
files = []
for name in ["main.tex", "main.bbl", "references.bib", "iclr2027_conference.sty",
             "iclr2027_conference.bst", "math_commands.tex"]:
    files.append((PAPER / name, name))
for folder, extension in [("figures", "*.pdf"), ("figures", "*.jpg"), ("tables", "*.tex")]:
    files.extend((path, str(path.relative_to(PAPER))) for path in sorted((PAPER / folder).glob(extension)))
for name in ["make_results.py", "make_figures.py", "build.sh", "package_source.py"]:
    files.append((ROOT / "scripts" / name, "scripts/" + name))
for name in ["reported_measurements.json", "results_ledger.json", "method_figure_source.json"]:
    files.append((ROOT / "evidence" / name, "evidence/" + name))
readme = ROOT / "evidence/README_source.md"
files.append((readme if readme.is_file() else ROOT / "README.md", "README.md"))
for path, _ in files:
    assert path.is_file(), path
    if path.suffix in {".tex", ".bib", ".json", ".md", ".py", ".sh"}:
        personal_prefix = "/" + "Users" + "/"
        assert personal_prefix not in path.read_text(), f"Personal absolute path found in {path.name}"
with ZipFile(target, "w", compression=ZIP_DEFLATED) as archive:
    for path, name in files:
        archive.write(path, name)
with ZipFile(target) as archive:
    assert archive.testzip() is None
print(f"Packaged {len(files)} files: {target.name}")
