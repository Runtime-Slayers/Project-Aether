"""
Package project code, notebooks, and `outputs/` artifacts into `release/gnesai_release.zip`.
"""
import os
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'release'
OUT.mkdir(exist_ok=True)

def add_folder_to_zip(z, folder: Path, arc_root: str):
    for p in folder.rglob('*'):
        if p.is_file():
            rel = p.relative_to(ROOT)
            z.write(p, arcname=str(rel))

def main():
    zip_path = OUT / 'gnesai_release.zip'
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
        # include source folders
        for d in ['data_factory', 'perception_layer', 'cognitive_core', 'neuro_symbolic', 'training', 'notebooks']:
            folder = ROOT / d
            if folder.exists():
                add_folder_to_zip(z, folder, d)
        # include outputs
        outputs = ROOT / 'outputs'
        if outputs.exists():
            add_folder_to_zip(z, outputs, 'outputs')
        # include README and requirements
        for f in ['README.md', 'requirements.txt']:
            p = ROOT / f
            if p.exists():
                z.write(p, arcname=f)

    print('Created release:', zip_path)

if __name__ == '__main__':
    main()
