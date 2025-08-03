"""
Example usage scenarios for ShapeNet Vector Database.
"""

import numpy as np
import os
from shapenet_vectordb import create_shapenet_db
from shapenet_utils import ShapeNetDBUtils, benchmark_search_performance


def example_basic_usage():
    """Basic usage example."""
    print("=" * 60)
    print("EXAMPLE 1: Basic Usage")
    print("=" * 60)
    
    # Create database
    db = create_shapenet_db(
        workspace="./example_db",
        db_type="hnsw",
        space='cosine'
    )
    
    # Index embeddings (small subset for example)
    print("Indexing embeddings...")
    db.index_embeddings(batch_size=20, max_files=50)
    
    # Get a shape to use as query
    shape_ids = db.list_shape_ids(limit=1)
    if shape_ids:
        query_shape_id = shape_ids[0]
        print(f"\nUsing '{query_shape_id}' as query...")
        
        # Get its embedding
        shape_data = db.search_by_shape_id(query_shape_id)
        if shape_data:
            # Search for similar shapes
            results = db.search_similar(
                shape_data['embedding'],
                limit=5,
                return_point_clouds=True
            )
            
            print(f"Found {len(results)} similar shapes:")
            for i, result in enumerate(results, 1):
                print(f"  {i}. {result['shape_id']} (similarity: {result['score']:.4f})")
                if result.get('point_cloud') is not None:
                    print(f"     Point cloud shape: {result['point_cloud'].shape}")


def example_batch_search():
    """Example of batch similarity search."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Batch Similarity Search")
    print("=" * 60)
    
    # Create database
    db = create_shapenet_db(workspace="./example_db")
    
    if not db.is_indexed:
        print("Database not indexed. Run basic example first.")
        return
    
    # Get multiple embeddings for batch search
    shape_ids = db.list_shape_ids(limit=3)
    query_embeddings = []
    
    for shape_id in shape_ids:
        shape_data = db.search_by_shape_id(shape_id)
        if shape_data:
            query_embeddings.append(shape_data['embedding'])
    
    if query_embeddings:
        print(f"Performing batch search with {len(query_embeddings)} queries...")
        
        utils = ShapeNetDBUtils(db)
        batch_results = utils.batch_similarity_search(query_embeddings, limit=3)
        
        for i, results in enumerate(batch_results):
            print(f"\nQuery {i+1} results:")
            for j, result in enumerate(results, 1):
                print(f"  {j}. {result['shape_id']} (score: {result['score']:.4f})")


def example_custom_embedding_query():
    """Example using custom embedding as query."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Custom Embedding Query")
    print("=" * 60)
    
    # Create database
    db = create_shapenet_db(workspace="./example_db")
    
    if not db.is_indexed:
        print("Database not indexed. Run basic example first.")
        return
    
    # Create a custom query embedding
    # In practice, this would come from your model processing an image/3D shape
    print("Creating custom query embedding...")
    
    # Option 1: Random embedding (for demonstration)
    custom_embedding = np.random.randn(1280).astype(np.float32)
    
    # Option 2: You could also modify an existing embedding
    shape_ids = db.list_shape_ids(limit=1)
    if shape_ids:
        shape_data = db.search_by_shape_id(shape_ids[0])
        if shape_data:
            # Add some noise to an existing embedding
            custom_embedding = shape_data['embedding'] + 0.1 * np.random.randn(1280)
    
    # Search with custom embedding
    print("Searching with custom embedding...")
    results = db.search_similar(custom_embedding, limit=5)
    
    print(f"Found {len(results)} results:")
    for i, result in enumerate(results, 1):
        print(f"  {i}. {result['shape_id']} (score: {result['score']:.4f})")


def example_shape_id_search():
    """Example of searching by specific shape ID."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Shape ID Search")
    print("=" * 60)
    
    # Create database
    db = create_shapenet_db(workspace="./example_db")
    
    if not db.is_indexed:
        print("Database not indexed. Run basic example first.")
        return
    
    # List available shapes
    shape_ids = db.list_shape_ids(limit=10)
    print("Available shape IDs (first 10):")
    for i, shape_id in enumerate(shape_ids, 1):
        print(f"  {i}. {shape_id}")
    
    if shape_ids:
        # Search for specific shape
        target_shape_id = shape_ids[0]
        print(f"\nSearching for shape: {target_shape_id}")
        
        shape_data = db.search_by_shape_id(target_shape_id)
        if shape_data:
            print(f"Found shape '{target_shape_id}':")
            print(f"  Point cloud shape: {shape_data['point_cloud'].shape}")
            print(f"  Embedding shape: {shape_data['embedding'].shape}")
            print(f"  Point cloud path: {shape_data['pc_path']}")
            print(f"  Embedding path: {shape_data['embedding_path']}")
        else:
            print(f"Shape '{target_shape_id}' not found")


def example_database_statistics():
    """Example showing database statistics."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Database Statistics")
    print("=" * 60)
    
    # Create database
    db = create_shapenet_db(workspace="./example_db")
    
    if not db.is_indexed:
        print("Database not indexed. Run basic example first.")
        return
    
    # Get basic stats
    stats = db.get_stats()
    print("Basic Database Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Get detailed embedding statistics
    utils = ShapeNetDBUtils(db)
    emb_stats = utils.get_embedding_statistics()
    print("\nEmbedding Statistics:")
    for key, value in emb_stats.items():
        if key not in ['mean_values', 'std_values']:
            print(f"  {key}: {value}")


def example_performance_benchmark():
    """Example performance benchmark."""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Performance Benchmark")
    print("=" * 60)
    
    # Create database
    db = create_shapenet_db(workspace="./example_db")
    
    if not db.is_indexed:
        print("Database not indexed. Run basic example first.")
        return
    
    # Run performance benchmark
    print("Running performance benchmark...")
    perf_results = benchmark_search_performance(
        db,
        num_queries=5,
        search_limits=[1, 5, 10]
    )
    
    print("\nPerformance Results:")
    for limit, metrics in perf_results["performance"].items():
        print(f"  Search limit {limit}:")
        print(f"    Average time: {metrics['avg_time']:.4f}s")
        print(f"    Min time: {metrics['min_time']:.4f}s")
        print(f"    Max time: {metrics['max_time']:.4f}s")


def example_export_results():
    """Example of exporting search results."""
    print("\n" + "=" * 60)
    print("EXAMPLE 7: Export Search Results")
    print("=" * 60)
    
    # Create database
    db = create_shapenet_db(workspace="./example_db")
    
    if not db.is_indexed:
        print("Database not indexed. Run basic example first.")
        return
    
    # Perform a search
    shape_ids = db.list_shape_ids(limit=1)
    if shape_ids:
        shape_data = db.search_by_shape_id(shape_ids[0])
        if shape_data:
            results = db.search_similar(shape_data['embedding'], limit=5)
            
            # Export results
            utils = ShapeNetDBUtils(db)
            output_file = "search_results_example.json"
            utils.export_search_results(results, output_file, include_embeddings=False)
            
            print(f"Search results exported to: {output_file}")
            
            # Also create an embedding index
            utils.create_embedding_index_file("embedding_index_example.json")
            print("Embedding index created: embedding_index_example.json")


def run_all_examples():
    """Run all examples in sequence."""
    try:
        example_basic_usage()
        example_batch_search()
        example_custom_embedding_query()
        example_shape_id_search()
        example_database_statistics()
        example_performance_benchmark()
        example_export_results()
        
        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        print("Make sure the shapenet directories exist and contain data.")


if __name__ == "__main__":
    # Check if shapenet directories exist
    if not os.path.exists("shapenet/shapenet_embedding"):
        print("Error: shapenet/shapenet_embedding directory not found!")
        print("Please make sure your shapenet data is properly set up.")
        exit(1)
    
    if not os.path.exists("shapenet/shapenet_pc"):
        print("Error: shapenet/shapenet_pc directory not found!")
        print("Please make sure your shapenet data is properly set up.")
        exit(1)
    
    # Run all examples
    run_all_examples()
