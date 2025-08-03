"""
Text Search Examples for ShapeNet Vector Database.
"""

import os
import numpy as np
from shapenet_vectordb import create_shapenet_db
from text_search import create_text_search, demo_text_search_queries, COMMON_SHAPE_QUERIES


def example_basic_text_search():
    """Basic text search example."""
    print("=" * 60)
    print("EXAMPLE 1: Basic Text Search")
    print("=" * 60)
    
    # Create and index vector database
    print("Setting up vector database...")
    db = create_shapenet_db(
        workspace="./text_search_db",
        db_type="hnsw",
        space='cosine'
    )
    
    # Index some embeddings
    db.index_embeddings(batch_size=50, max_files=200)  # Index subset for demo
    print(f"Indexed {db.get_stats()['total_indexed']} shapes")
    
    # Create text search
    print("\nLoading CLIP model for text search...")
    text_search = create_text_search(db)
    
    # Print model info
    model_info = text_search.get_model_info()
    print(f"Model info: {model_info}")
    
    # Perform text searches
    test_queries = [
        "wooden chair",
        "round table",
        "car with four wheels",
        "airplane with wings"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Searching for: '{query}'")
        try:
            results = text_search.search_by_text(query, limit=3)
            
            if results:
                print(f"Found {len(results)} results:")
                for i, result in enumerate(results, 1):
                    print(f"  {i}. {result['shape_id']}")
                    print(f"     Similarity: {result['score']:.4f}")
                    print(f"     Path: {result['pc_path']}")
            else:
                print("No results found")
                
        except Exception as e:
            print(f"Error searching for '{query}': {e}")


def example_batch_text_search():
    """Example of batch text search."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Batch Text Search")
    print("=" * 60)
    
    # Use existing database
    db = create_shapenet_db(workspace="./text_search_db")
    
    if not db.is_indexed:
        print("Database not indexed. Run basic example first.")
        return
    
    # Create text search
    text_search = create_text_search(db)
    
    # Batch search with demo queries
    batch_queries = demo_text_search_queries()[:4]  # Use first 4 queries
    
    print(f"Performing batch search for {len(batch_queries)} queries...")
    batch_results = text_search.batch_text_search(batch_queries, limit=2)
    
    for i, (query, results) in enumerate(zip(batch_queries, batch_results)):
        print(f"\nQuery {i+1}: '{query}'")
        for j, result in enumerate(results, 1):
            print(f"  {j}. {result['shape_id']} (score: {result['score']:.4f})")


def example_text_similarity_comparison():
    """Example of comparing text similarities."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Text Similarity Comparison")
    print("=" * 60)
    
    # Use existing database
    db = create_shapenet_db(workspace="./text_search_db")
    text_search = create_text_search(db)
    
    # Compare different text descriptions
    text_pairs = [
        ("chair", "seat"),
        ("table", "desk"),
        ("car", "automobile"),
        ("airplane", "vehicle"),
        ("chair", "airplane"),  # Different objects
    ]
    
    print("Comparing text similarities:")
    for text1, text2 in text_pairs:
        similarity = text_search.compare_text_embeddings(text1, text2)
        print(f"'{text1}' vs '{text2}': {similarity:.4f}")


def example_detailed_search_with_point_clouds():
    """Example showing detailed search results with point clouds."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Detailed Search with Point Clouds")
    print("=" * 60)
    
    # Use existing database
    db = create_shapenet_db(workspace="./text_search_db")
    
    if not db.is_indexed:
        print("Database not indexed. Run basic example first.")
        return
    
    text_search = create_text_search(db)
    
    # Search with point cloud loading
    query = "chair"
    print(f"🔍 Detailed search for: '{query}'")
    
    results = text_search.search_by_text(
        query, 
        limit=3, 
        return_point_clouds=True
    )
    
    for i, result in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"  Shape ID: {result['shape_id']}")
        print(f"  Similarity Score: {result['score']:.4f}")
        print(f"  Point Cloud Path: {result['pc_path']}")
        
        if 'point_cloud' in result and result['point_cloud'] is not None:
            pc = result['point_cloud']
            print(f"  Point Cloud Shape: {pc.shape}")
            print(f"  Point Cloud Stats:")
            print(f"    Min: [{pc.min(axis=0)}]")
            print(f"    Max: [{pc.max(axis=0)}]")
            print(f"    Mean: [{pc.mean(axis=0)}]")
        else:
            print("  Point Cloud: Not loaded")


def example_comprehensive_search():
    """Comprehensive search example with various queries."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Comprehensive Search")
    print("=" * 60)
    
    # Use existing database
    db = create_shapenet_db(workspace="./text_search_db")
    
    if not db.is_indexed:
        print("Database not indexed. Run basic example first.")
        return
    
    text_search = create_text_search(db)
    
    # Test various types of queries
    query_categories = {
        "Furniture": ["chair", "table", "sofa", "bed"],
        "Vehicles": ["car", "airplane", "bicycle"],
        "Electronics": ["computer", "phone", "television"],
        "Abstract": ["round object", "long thin item", "curved shape"]
    }
    
    for category, queries in query_categories.items():
        print(f"\n📂 {category} Queries:")
        for query in queries:
            try:
                results = text_search.search_by_text(query, limit=1)
                if results:
                    best_result = results[0]
                    print(f"  '{query}' → {best_result['shape_id']} (score: {best_result['score']:.4f})")
                else:
                    print(f"  '{query}' → No results")
            except Exception as e:
                print(f"  '{query}' → Error: {e}")


def interactive_text_search():
    """Interactive text search demo."""
    print("\n" + "=" * 60)
    print("INTERACTIVE TEXT SEARCH")
    print("=" * 60)
    
    # Use existing database
    db = create_shapenet_db(workspace="./text_search_db")
    
    if not db.is_indexed:
        print("Database not indexed. Run basic example first.")
        return
    
    text_search = create_text_search(db)
    
    print("Enter text descriptions to search for shapes.")
    print("Commands:")
    print("  - Type any description (e.g., 'red chair', 'round table')")
    print("  - 'quit' to exit")
    print("  - 'stats' to show database stats")
    print()
    
    while True:
        try:
            query = input("Search query: ").strip()
            
            if query.lower() == 'quit':
                print("Goodbye!")
                break
            elif query.lower() == 'stats':
                stats = db.get_stats()
                print(f"Database stats: {stats}")
                continue
            elif not query:
                continue
            
            print(f"🔍 Searching for: '{query}'")
            results = text_search.search_by_text(query, limit=5)
            
            if results:
                print(f"Found {len(results)} results:")
                for i, result in enumerate(results, 1):
                    print(f"  {i}. {result['shape_id']} (similarity: {result['score']:.4f})")
            else:
                print("No results found")
            
            print()
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def run_all_text_search_examples():
    """Run all text search examples."""
    print("🚀 ShapeNet Text Search Examples")
    print("=" * 60)
    
    # Check if CLIP model exists
    model_path = "models/open_clip_pytorch_model.bin"
    if not os.path.exists(model_path):
        print(f"❌ CLIP model not found at: {model_path}")
        print("Please make sure the CLIP model file is available.")
        return
    
    try:
        example_basic_text_search()
        example_batch_text_search()
        example_text_similarity_comparison()
        example_detailed_search_with_point_clouds()
        example_comprehensive_search()
        
        print("\n" + "=" * 60)
        print("✅ All text search examples completed!")
        print("\nTo try interactive search, run:")
        print("  python text_search_examples.py --interactive")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")


if __name__ == "__main__":
    import sys
    
    if "--interactive" in sys.argv:
        interactive_text_search()
    else:
        run_all_text_search_examples()
