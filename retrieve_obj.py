import zipfile
import os
from pathlib import Path
import shutil

def retrieve_shapenet_model(shape_id: str, 
                            output_dir: str = "./retrieved", 
                            zip_root_dir: str = "hf_shapenet_zips"):
    """
    Extract a ShapeNetCore model folder from a ZIP file without full extraction.

    Args:
        shape_id (str): Shape ID in the form "synset_id-object_id".
        output_dir (str or Path): Destination directory to extract files to.
        zip_root_dir (str or Path): Directory containing <synset_id>.zip files.
    """
    synset_id, object_id = shape_id.split("-")
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
                # Compute the relative path preserving the full structure
                rel_path = Path(member)
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
    zip_path = "./hf_shapenet_zips"
    shape_id = "02691156-b089abdb33c39321afd477f714c68df9"  # synset_id-object_id
    output_dir = "./shapenet_models"

    retrieve_shapenet_model(
        shape_id=shape_id, 
        output_dir=output_dir, 
        zip_root_dir=zip_path
    )
