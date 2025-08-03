import numpy as np
from shapenet_vectordb import create_shapenet_db


def main():
    print("ShapeNet Vector Database Demo")
    print("=" * 40)
    
    # Create the vector database
    print("Creating ShapeNet vector database...")
    db = create_shapenet_db(
        workspace="./vectordb_workspace",
        db_type="hnsw",  # Use HNSW for faster approximate search
        pc_dir="shapenet/shapenet_pc",
        embedding_dir="shapenet/shapenet_embedding",
        space='cosine',  # Use cosine similarity
        max_elements=50000  # Set max elements for your dataset
    )
    
    print(f"Database created successfully!")
    print(f"Database stats: {db.get_stats()}")
    
    # Index a subset of embeddings for demonstration (limit to 100 for quick testing)
    print("\nIndexing embeddings (limited to 100 for demo)...")
    try:
        db.index_embeddings(batch_size=50, max_files=100)
        print("Indexing completed!")
        
        # Show database stats after indexing
        stats = db.get_stats()
        print(f"\nDatabase stats after indexing:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Demonstrate similarity search
        print("\nDemonstrating similarity search...")
        
        # Get a list of shape IDs to use for testing
        shape_ids = db.list_shape_ids(limit=5)
        if shape_ids:
            test_shape_id = shape_ids[0]
            print(f"Using shape '{test_shape_id}' as query...")
            
            # Get the embedding for this shape as a query
            shape_data = db.search_by_shape_id(test_shape_id)
            if shape_data:
                query_embedding = shape_data['embedding']
                
                # Search for similar shapes
                print(f"Searching for shapes similar to '{test_shape_id}'...")
                similar_shapes = db.search_similar(
                    query_embedding=query_embedding,
                    limit=5,
                    return_point_clouds=False  # Set to True if you want point cloud data
                )
                
                print(f"\nFound {len(similar_shapes)} similar shapes:")
                for i, result in enumerate(similar_shapes, 1):
                    print(f"  {i}. Shape ID: {result['shape_id']}")
                    print(f"     Similarity Score: {result['score']:.4f}")
                    print(f"     Point Cloud Path: {result['pc_path']}")
                    print()
                
                # Demonstrate point search
                print("Demonstrating point search...")
                if len(shape_ids) > 1:
                    search_shape_id = shape_ids[1]
                    print(f"Searching for specific shape: '{search_shape_id}'")
                    
                    point_result = db.search_by_shape_id(search_shape_id)
                    if point_result:
                        print(f"Found shape '{search_shape_id}':")
                        print(f"  Point cloud shape: {point_result['point_cloud'].shape}")
                        print(f"  Embedding shape: {point_result['embedding'].shape}")
                        print(f"  Point cloud path: {point_result['pc_path']}")
                    else:
                        print(f"Shape '{search_shape_id}' not found")
            else:
                print(f"Could not load data for shape '{test_shape_id}'")
        else:
            print("No shape IDs available for testing")
            
    except Exception as e:
        print(f"Error during operation: {e}")
        print("This might be because the shapenet directories are not accessible.")
        print("Make sure the 'shapenet/shapenet_pc' and 'shapenet/shapenet_embedding' directories exist.")


def demo_custom_query():
    """
    Demonstrate how to use the database with a custom embedding query.
    """
    print("\n" + "=" * 40)
    print("Custom Query Demo")
    print("=" * 40)
    
    # Create database
    db = create_shapenet_db()
    
    if not db.is_indexed:
        print("Database needs to be indexed first. Run the main demo.")
        return
    
    # Create a random query embedding (in practice, this would come from your model)
    random_query = np.random.randn(1280).astype(np.float32)
    
    print("Searching with random query embedding...")
    results = db.search_similar(
        query_embedding=random_query,
        limit=3,
        return_point_clouds=True
    )
    
    print(f"Found {len(results)} results:")
    for i, result in enumerate(results, 1):
        print(f"  {i}. Shape: {result['shape_id']}")
        print(f"     Score: {result['score']:.4f}")
        if result.get('point_cloud') is not None:
            print(f"     Point cloud shape: {result['point_cloud'].shape}")


if __name__ == "__main__":
    main()
    
    # Uncomment the line below to run the custom query demo
    # demo_custom_query()
