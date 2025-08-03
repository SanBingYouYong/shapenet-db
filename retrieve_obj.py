import zipfile
import os
from pathlib import Path
import shutil

def extract_object_from_shapenetcore(zip_root_dir, synset_id, object_id, output_dir):
    """
    Extract a ShapeNetCore model folder from a ZIP file without full extraction.

    Args:
        zip_root_dir (str or Path): Directory containing <synset_id>.zip files.
        synset_id (str): Category ID (e.g., '03001627' for chairs).
        object_id (str): Object ID (e.g., '1a04e3eab45ca15dd86060f189eb133').
        output_dir (str or Path): Destination directory to extract files to.
    """
    zip_path = Path(zip_root_dir).expanduser() / f"{synset_id}.zip"
    output_dir = Path(output_dir).expanduser()
    folder_path = f"{synset_id}/{object_id}/"

    if not zip_path.exists():
        raise FileNotFoundError(f"ZIP file not found: {zip_path}")

    with zipfile.ZipFile(zip_path, 'r') as zipf:
        # Check if the folder exists in the zip
        if folder_path not in zipf.namelist():
            raise FileNotFoundError(f"Folder {folder_path} not found in ZIP {zip_path}")

        # Copy all files and subfolders under the folder_path
        for member in zipf.namelist():
            if member.startswith(folder_path) and member != folder_path:
                # Compute the relative path inside the object folder
                rel_path = Path(member).relative_to(synset_id)
                target_path = output_dir / rel_path
                if member.endswith('/'):
                    target_path.mkdir(parents=True, exist_ok=True)
                else:
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    with zipf.open(member) as source, open(target_path, 'wb') as target:
                        shutil.copyfileobj(source, target)
        print(f"✅ Extracted {folder_path} to {output_dir / synset_id / object_id}")

# === Example usage ===
if __name__ == "__main__":
    zip_path = "~/.cache/huggingface/hub/datasets--ShapeNet--ShapeNetCore/snapshots/0efb24cbe6828a85771a28335c5f7b5626514d9b/"
    synset_id = "02691156"  # e.g., chairs
    object_id = "b089abdb33c39321afd477f714c68df9"  # a specific chair
    output_dir = "./shapenet_models"

    extract_object_from_shapenetcore(zip_path, synset_id, object_id, output_dir)
