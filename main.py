import numpy as np
from shapenet_vectordb import create_shapenet_db
from text_search import create_text_search


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
                    score_str = f"{result['score']:.4f}" if result['score'] is not None else "N/A"
                    print(f"  {i}. Shape ID: {result['shape_id']}")
                    print(f"     Similarity Score: {score_str}")
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
                
                # Demonstrate text search
                print("\n" + "=" * 40)
                print("Demonstrating Text Search")
                print("=" * 40)
                
                try:
                    # Check if CLIP model exists
                    import os
                    model_path = "models/open_clip_pytorch_model.bin"
                    if os.path.exists(model_path):
                        print("Loading CLIP model for text search...")
                        text_search = create_text_search(db)
                        
                        # Test text queries
                        test_queries = [
                            "chair",
                            "table",
                            "car with wheels"
                        ]
                        
                        for query in test_queries:
                            print(f"\n🔍 Text search for: '{query}'")
                            try:
                                text_results = text_search.search_by_text(query, limit=3)
                                
                                if text_results:
                                    print(f"Found {len(text_results)} results:")
                                    for i, result in enumerate(text_results, 1):
                                        score_str = f"{result['score']:.4f}" if result['score'] is not None else "N/A"
                                        print(f"  {i}. {result['shape_id']} (similarity: {score_str})")
                                else:
                                    print("No results found")
                            except Exception as e:
                                print(f"Error in text search: {e}")
                        
                        print("\n✅ Text search working! You can now search using natural language.")
                        
                    else:
                        print(f"CLIP model not found at: {model_path}")
                        print("Text search requires the CLIP model to be available.")
                        print("The vector database still works for embedding-based search.")
                        
                except Exception as e:
                    print(f"Error setting up text search: {e}")
                    print("Vector database still works for embedding-based search.")
                
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
        score_str = f"{result['score']:.4f}" if result['score'] is not None else "N/A"
        print(f"  {i}. Shape: {result['shape_id']}")
        print(f"     Score: {score_str}")
        if result.get('point_cloud') is not None:
            print(f"     Point cloud shape: {result['point_cloud'].shape}")


def demo_text_search():
    """
    Demonstrate text search functionality.
    """
    print("\n" + "=" * 40)
    print("Text Search Demo")
    print("=" * 40)
    
    # Create database
    db = create_shapenet_db()
    
    if not db.is_indexed:
        print("Database needs to be indexed first. Run the main demo.")
        return
    
    try:
        # Create text search
        text_search = create_text_search(db)
        
        # Interactive text search
        print("Interactive text search - enter descriptions to search for shapes:")
        print("Type 'quit' to exit")
        
        while True:
            query = input("\nEnter text description: ").strip()
            
            if query.lower() == 'quit':
                break
            
            if not query:
                continue
            
            try:
                results = text_search.search_by_text(query, limit=3)
                
                if results:
                    print(f"Found {len(results)} results for '{query}':")
                    for i, result in enumerate(results, 1):
                        score_str = f"{result['score']:.4f}" if result['score'] is not None else "N/A"
                        print(f"  {i}. {result['shape_id']} (similarity: {score_str})")
                else:
                    print("No results found")
                    
            except Exception as e:
                print(f"Error: {e}")
    
    except Exception as e:
        print(f"Error setting up text search: {e}")


if __name__ == "__main__":
    main()
    
    # Uncomment the lines below to run additional demos
    # demo_custom_query()
    # demo_text_search()
