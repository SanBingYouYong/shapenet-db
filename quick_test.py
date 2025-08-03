"""
Quick test script to verify the ShapeNet Vector Database functionality.
"""

import os
import numpy as np
from shapenet_vectordb import create_shapenet_db

def quick_test():
    """Quick test of the vector database functionality."""
    print("ShapeNet Vector Database - Quick Test")
    print("=" * 40)
    
    # Check if shapenet directories exist
    if not os.path.exists("shapenet/shapenet_embedding"):
        print("❌ shapenet/shapenet_embedding directory not found!")
        return False
    
    if not os.path.exists("shapenet/shapenet_pc"):
        print("❌ shapenet/shapenet_pc directory not found!")
        return False
    
    print("✅ ShapeNet directories found")
    
    try:
        # Create database
        print("\n1. Creating vector database...")
        db = create_shapenet_db(
            workspace="./test_vectordb",
            db_type="hnsw",
            space='cosine',
            max_elements=1000  # Small for testing
        )
        print("✅ Database created successfully")
        
        # Index a small subset for testing
        print("\n2. Indexing embeddings (limited to 10 for quick test)...")
        db.index_embeddings(batch_size=5, max_files=10)
        print("✅ Indexing completed")
        
        # Show stats
        stats = db.get_stats()
        print(f"\n3. Database stats:")
        print(f"   Total indexed: {stats['total_indexed']}")
        print(f"   Is indexed: {stats['is_indexed']}")
        
        if stats['total_indexed'] > 0:
            # Test similarity search
            print("\n4. Testing similarity search...")
            shape_ids = db.list_shape_ids(limit=3)
            print(f"   Available shapes: {shape_ids}")
            
            if shape_ids:
                # Use first shape as query
                test_shape_id = shape_ids[0]
                shape_data = db.search_by_shape_id(test_shape_id)
                
                if shape_data:
                    print(f"   Using '{test_shape_id}' as query...")
                    print(f"   Point cloud shape: {shape_data['point_cloud'].shape}")
                    print(f"   Embedding shape: {shape_data['embedding'].shape}")
                    
                    # Search for similar shapes
                    results = db.search_similar(
                        shape_data['embedding'],
                        limit=3,
                        return_point_clouds=False
                    )
                    
                    print(f"   Found {len(results)} similar shapes:")
                    for i, result in enumerate(results, 1):
                        score_str = f"{result['score']:.4f}" if result['score'] is not None else "N/A"
                        print(f"     {i}. {result['shape_id']} (score: {score_str})")
                    
                    print("✅ Similarity search working")
                    
                    # Test custom embedding query
                    print("\n5. Testing custom embedding query...")
                    custom_embedding = np.random.randn(1280).astype(np.float32)
                    custom_results = db.search_similar(custom_embedding, limit=2)
                    print(f"   Random embedding found {len(custom_results)} results:")
                    for i, result in enumerate(custom_results, 1):
                        score_str = f"{result['score']:.4f}" if result['score'] is not None else "N/A"
                        print(f"     {i}. {result['shape_id']} (score: {score_str})")
                    
                    print("✅ Custom embedding search working")
                    
                else:
                    print(f"❌ Could not load data for shape '{test_shape_id}'")
                    return False
            else:
                print("❌ No shape IDs available")
                return False
        else:
            print("❌ No shapes were indexed")
            return False
        
        print("\n" + "=" * 40)
        print("🎉 All tests passed! Vector database is working correctly.")
        print("\nYou can now:")
        print("- Run 'python main.py' for a full demo")
        print("- Run 'python examples.py' for comprehensive examples")
        print("- Use the database in your own scripts")
        return True
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        return False

if __name__ == "__main__":
    success = quick_test()
    if not success:
        print("\n💡 If you see errors, make sure:")
        print("   - The shapenet directories contain .npy files")
        print("   - The embedding files have '_embedding' suffix")
        print("   - Point cloud files have corresponding names")
