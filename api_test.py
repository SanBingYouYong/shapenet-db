import requests

BASE_URL = "http://localhost:8001"

def search_shapes(query, limit=1, method="GET"):
    """Search ShapeNet models by text."""
    if method.upper() == "GET":
        resp = requests.get(f"{BASE_URL}/search", params={"query": query, "limit": limit})
    else:  # POST
        resp = requests.post(f"{BASE_URL}/search", json={"query": query, "limit": limit})
    
    resp.raise_for_status()
    return resp.json()

def search_and_download_glb(query, limit=1, output_path="result.glb"):
    """Search ShapeNet and download the best match as a GLB file."""
    resp = requests.get(
        f"{BASE_URL}/search-glb",
        params={"query": query, "limit": limit}
    )
    resp.raise_for_status()
    
    with open(output_path, "wb") as f:
        f.write(resp.content)
    
    print(f"GLB file saved to: {output_path}")
    print(f"Shape ID: {resp.headers.get('X-Shape-ID')}")
    print(f"Score: {resp.headers.get('X-Score')}")


if __name__ == "__main__":
    # Example: Text search
    results = search_shapes("a modern chair", limit=3, method="GET")
    print("Search results:", results)

    # Example: Search and download GLB
    search_and_download_glb("a wooden table", limit=1, output_path="table.glb")
