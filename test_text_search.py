"""
Quick test for text search functionality.
"""

import os
import numpy as np
from shapenet_vectordb import create_shapenet_db
from text_search import create_text_search


def test_text_search():
    """Test the text search functionality."""
    print("Text Search Test")
    print("=" * 30)
    
    # Check if CLIP model exists
    model_path = "models/open_clip_pytorch_model.bin"
    if not os.path.exists(model_path):
        print(f"❌ CLIP model not found at: {model_path}")
        print("Please ensure the CLIP model file is available.")
        return False
    
    print("✅ CLIP model found")
    
    try:
        # Create and index vector database (small subset for testing)
        print("Setting up vector database...")
        db = create_shapenet_db(
            workspace="./test_text_db",
            db_type="hnsw",
            space='cosine'
        )
        
        # Index a small number of embeddings for quick testing
        db.index_embeddings(batch_size=10, max_files=20)
        print(f"✅ Indexed {db.get_stats()['total_indexed']} shapes")
        
        # Create text search
        print("Loading CLIP model...")
        text_search = create_text_search(db)
        print("✅ Text search initialized")
        
        # Test model info
        model_info = text_search.get_model_info()
        print(f"Model: {model_info['model_name']} on {model_info['device']}")
        
        # Test text encoding
        print("\nTesting text encoding...")
        test_text = "chair"
        embedding = text_search.encode_text(test_text)
        print(f"✅ Text '{test_text}' encoded to shape: {embedding.shape}")
        
        # Test text search
        print("\nTesting text search...")
        results = text_search.search_by_text("chair", limit=3)
        
        if results:
            print(f"✅ Text search successful! Found {len(results)} results:")
            for i, result in enumerate(results, 1):
                score_str = f"{result['score']:.4f}" if result['score'] is not None else "N/A"
                print(f"  {i}. {result['shape_id']} (score: {score_str})")
        else:
            print("⚠️  No results found, but search executed successfully")
        
        # Test text similarity
        print("\nTesting text similarity...")
        similarity = text_search.compare_text_embeddings("chair", "seat")
        print(f"✅ Similarity between 'chair' and 'seat': {similarity:.4f}")
        
        print("\n🎉 All text search tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error during text search test: {e}")
        return False


if __name__ == "__main__":
    success = test_text_search()
    
    if success:
        print("\n💡 Text search is ready! You can now:")
        print("  - Run 'python main.py' for full demo with text search")
        print("  - Run 'python text_search_examples.py' for comprehensive examples")
        print("  - Use text_search.search_by_text('your query') in your code")
    else:
        print("\n💡 Make sure:")
        print("  - The CLIP model file is in models/open_clip_pytorch_model.bin")
        print("  - The shapenet directories contain embeddings")
        print("  - All dependencies are installed (open-clip-torch, torch, etc.)")
