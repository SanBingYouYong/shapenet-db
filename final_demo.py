"""
Comprehensive ShapeNet Vector Database Demo
"""

import os
import numpy as np
from shapenet_vectordb import create_shapenet_db
from text_search import create_text_search
from shapenet_utils import ShapeNetDBUtils, ShapeNetTextSearchUtils


def comprehensive_demo():
    """Comprehensive demonstration of all ShapeNet Vector Database features."""
    print("🎯 ShapeNet Vector Database - Comprehensive Demo")
    print("=" * 60)
    
    # 1. Setup Vector Database
    print("\n1️⃣ Setting up Vector Database...")
    db = create_shapenet_db(
        workspace="./comprehensive_demo_db",
        db_type="hnsw",
        space='cosine',
        max_elements=10000,
        ef_construction=200,
        ef=50,
        M=16
    )
    
    # Index a reasonable subset for demo
    print("Indexing embeddings (500 shapes for comprehensive demo)...")
    db.index_embeddings(batch_size=100, max_files=500)
    
    stats = db.get_stats()
    print(f"✅ Database ready with {stats['total_indexed']} indexed shapes")
    
    # 2. Setup Text Search
    print("\n2️⃣ Setting up Text Search...")
    if os.path.exists("models/open_clip_pytorch_model.bin"):
        text_search = create_text_search(db)
        model_info = text_search.get_model_info()
        print(f"✅ Text search ready - {model_info['model_name']} on {model_info['device']}")
        text_search_available = True
    else:
        print("⚠️ CLIP model not found - text search disabled")
        text_search_available = False
    
    # 3. Demonstrate Embedding-based Search
    print("\n3️⃣ Embedding-based Similarity Search...")
    shape_ids = db.list_shape_ids(limit=10)
    demo_shape_id = shape_ids[0]
    
    shape_data = db.search_by_shape_id(demo_shape_id)
    print(f"Using shape '{demo_shape_id}' as query")
    print(f"Point cloud shape: {shape_data['point_cloud'].shape}")
    
    similar_results = db.search_similar(shape_data['embedding'], limit=5)
    print(f"Found {len(similar_results)} similar shapes:")
    for i, result in enumerate(similar_results, 1):
        score = f"{result['score']:.4f}" if result['score'] is not None else "N/A"
        print(f"  {i}. {result['shape_id']} (score: {score})")
    
    # 4. Demonstrate Text Search
    if text_search_available:
        print("\n4️⃣ Text-based Search...")
        
        # Test various queries
        text_queries = [
            "wooden chair",
            "round dining table", 
            "red car",
            "airplane with wings",
            "computer laptop",
            "coffee cup"
        ]
        
        all_text_results = {}
        for query in text_queries:
            print(f"\n🔍 '{query}':")
            try:
                results = text_search.search_by_text(query, limit=3)
                all_text_results[query] = results
                
                if results:
                    for i, result in enumerate(results, 1):
                        score = f"{result['score']:.4f}" if result['score'] is not None else "N/A"
                        print(f"  {i}. {result['shape_id']} (score: {score})")
                else:
                    print("  No results found")
            except Exception as e:
                print(f"  Error: {e}")
        
        # Text similarity comparisons
        print("\n📊 Text Similarity Analysis:")
        comparisons = [
            ("chair", "seat"),
            ("table", "desk"), 
            ("car", "vehicle"),
            ("airplane", "plane"),
            ("laptop", "computer")
        ]
        
        for text1, text2 in comparisons:
            similarity = text_search.compare_text_embeddings(text1, text2)
            print(f"  '{text1}' ↔ '{text2}': {similarity:.4f}")
    
    # 5. Advanced Features
    print("\n5️⃣ Advanced Features...")
    
    # Database utilities
    utils = ShapeNetDBUtils(db)
    
    # Embedding statistics
    emb_stats = utils.get_embedding_statistics()
    if 'error' not in emb_stats:
        print(f"📈 Embedding Statistics:")
        print(f"  Samples analyzed: {emb_stats['num_samples']}")
        print(f"  Embedding dimension: {emb_stats['embedding_dim']}")
        print(f"  Mean norm: {emb_stats['mean_norm']:.4f}")
        print(f"  Std norm: {emb_stats['std_norm']:.4f}")
    
    # Pattern search
    print(f"\n🔍 Pattern Search Examples:")
    chair_shapes = utils.find_shape_by_pattern("*chair*")
    table_shapes = utils.find_shape_by_pattern("*table*")
    print(f"  Shapes with 'chair' in ID: {len(chair_shapes)}")
    print(f"  Shapes with 'table' in ID: {len(table_shapes)}")
    
    # 6. Export Examples
    print("\n6️⃣ Export Functionality...")
    
    # Export similarity search results
    export_results = db.search_similar(shape_data['embedding'], limit=10)
    utils.export_search_results(export_results, "demo_similarity_results.json")
    
    # Export text search results if available
    if text_search_available and all_text_results:
        text_utils = ShapeNetTextSearchUtils(text_search)
        text_utils.batch_text_search_with_export(
            list(all_text_results.keys())[:3],  # First 3 queries
            output_dir="./demo_text_results",
            limit=5
        )
    
    # Create embedding index
    utils.create_embedding_index_file("demo_embedding_index.json")
    
    # 7. Performance Summary
    print("\n7️⃣ Performance Summary...")
    print(f"📊 Database Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    if text_search_available:
        print(f"\n🔤 Text Search Capabilities:")
        print(f"  Model: {model_info['model_name']}")
        print(f"  Device: {model_info['device']}")
        print(f"  Embedding dimension: {model_info['embedding_dim']}")
        print(f"  Text queries tested: {len(text_queries)}")
    
    print("\n" + "=" * 60)
    print("🎉 Comprehensive Demo Complete!")
    print("=" * 60)
    
    print("\n📁 Generated Files:")
    print("  - demo_similarity_results.json")
    print("  - demo_embedding_index.json")
    if text_search_available:
        print("  - demo_text_results/ (directory with text search results)")
    
    print("\n🚀 What You Can Do Next:")
    print("  1. Use db.search_similar(embedding, limit=N) for embedding search")
    if text_search_available:
        print("  2. Use text_search.search_by_text('query', limit=N) for text search")
    print("  3. Use db.search_by_shape_id('shape_id') for direct lookup")
    print("  4. Scale up to your full 50k+ dataset")
    print("  5. Integrate into your own applications")
    
    return db, text_search if text_search_available else None


def interactive_demo():
    """Interactive demo allowing user to try different searches."""
    print("\n" + "=" * 60)
    print("🎮 Interactive Demo Mode")
    print("=" * 60)
    
    # Use existing database if available
    try:
        db = create_shapenet_db(workspace="./comprehensive_demo_db")
        if not db.is_indexed:
            print("No indexed database found. Please run the comprehensive demo first.")
            return
        
        if os.path.exists("models/open_clip_pytorch_model.bin"):
            text_search = create_text_search(db)
            print("✅ Text search available")
        else:
            text_search = None
            print("⚠️ Text search not available (CLIP model not found)")
        
        print("\nCommands:")
        print("  text: <description>  - Search by text description")
        print("  shape: <shape_id>    - Look up specific shape")
        print("  random               - Search with random embedding")
        print("  stats                - Show database statistics")
        print("  quit                 - Exit")
        
        while True:
            try:
                command = input("\n> ").strip()
                
                if command.lower() == 'quit':
                    break
                elif command.lower() == 'stats':
                    stats = db.get_stats()
                    for key, value in stats.items():
                        print(f"  {key}: {value}")
                elif command.lower() == 'random':
                    random_emb = np.random.randn(1280).astype(np.float32)
                    results = db.search_similar(random_emb, limit=3)
                    print(f"Random search found {len(results)} results:")
                    for i, result in enumerate(results, 1):
                        score = f"{result['score']:.4f}" if result['score'] is not None else "N/A"
                        print(f"  {i}. {result['shape_id']} (score: {score})")
                elif command.startswith('text:'):
                    if not text_search:
                        print("Text search not available")
                        continue
                    query = command[5:].strip()
                    if query:
                        results = text_search.search_by_text(query, limit=5)
                        print(f"Text search for '{query}' found {len(results)} results:")
                        for i, result in enumerate(results, 1):
                            score = f"{result['score']:.4f}" if result['score'] is not None else "N/A"
                            print(f"  {i}. {result['shape_id']} (score: {score})")
                elif command.startswith('shape:'):
                    shape_id = command[6:].strip()
                    if shape_id:
                        result = db.search_by_shape_id(shape_id)
                        if result:
                            print(f"Found shape '{shape_id}':")
                            print(f"  Point cloud: {result['point_cloud'].shape}")
                            print(f"  Embedding: {result['embedding'].shape}")
                        else:
                            print(f"Shape '{shape_id}' not found")
                else:
                    print("Unknown command. Type 'quit' to exit.")
                    
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    except Exception as e:
        print(f"Error setting up interactive demo: {e}")


if __name__ == "__main__":
    import sys
    
    if "--interactive" in sys.argv:
        interactive_demo()
    else:
        db, text_search = comprehensive_demo()
        
        # Ask if user wants to try interactive mode
        try:
            response = input("\nWould you like to try interactive mode? (y/n): ").strip().lower()
            if response in ['y', 'yes']:
                interactive_demo()
        except KeyboardInterrupt:
            print("\nGoodbye!")
