import bpy
import os


def obj_to_glb(obj_path: str, output_glb_path: str):
    """
    Convert an OBJ file to GLB format using Blender.
    
    Args:
        obj_path (str): Absolute path to the input OBJ file
        output_glb_path (str): Absolute path for the output GLB file
    
    Raises:
        ValueError: If either path is not absolute
    """
    # Validate that paths are absolute
    if not os.path.isabs(obj_path):
        raise ValueError(f"obj_path must be an absolute path, got: {obj_path}")
    if not os.path.isabs(output_glb_path):
        raise ValueError(f"output_glb_path must be an absolute path, got: {output_glb_path}")
    
    # Clear the current scene
    bpy.ops.wm.read_homefile(use_empty=True)
    
    # Import OBJ file
    bpy.ops.wm.obj_import(filepath=obj_path)
    
    # Export as GLB
    bpy.ops.export_scene.gltf(
        filepath=output_glb_path,
    )


# Example usage
if __name__ == "__main__":
    obj_path = os.path.abspath("retrieved/04379243/553c416f33c5e5e18b9b51ae4415d5aa/models/model_normalized.obj")
    output_glb_path = os.path.abspath("temp.glb")
    obj_to_glb(obj_path, output_glb_path)

