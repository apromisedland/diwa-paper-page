"""Package the current manuscript inputs without modifying the original archive."""
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

root = Path(__file__).resolve().parents[1]
source = root / "paper/DIWA__Decision_Influential_World_Abstraction_for_VLA_WAM_Policies"
target = root / "assets/diwa-source.zip"
files = [source / name for name in (
    "main.tex", "references.bib", "iclr2027_conference.sty",
    "iclr2027_conference.bst", "math_commands.tex", "README.md",
)]
for directory, patterns in {"figures": ("*.pdf", "*.jpg"), "tables": ("*.tex",),
                            "scripts": ("*.py", "*.sh"), "evidence": ("*.json", "*.md")}.items():
    for pattern in patterns:
        files.extend(sorted((source / directory).glob(pattern)))
for file in files:
    if not file.is_file():
        raise FileNotFoundError(file)
with ZipFile(target, "w", compression=ZIP_DEFLATED) as archive:
    for file in files:
        archive.write(file, file.relative_to(source).as_posix())
with ZipFile(target) as archive:
    assert archive.testzip() is None
    assert archive.read("main.tex") == (source / "main.tex").read_bytes()
print(f"Packaged {len(files)} current manuscript files into {target.name}")
