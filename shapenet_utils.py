"""
Utility functions for ShapeNet Vector Database operations.
"""

import os
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
import json
from pathlib import Path

from shapenet_vectordb import ShapeNetVectorDB, create_shapenet_db


class ShapeNetDBUtils:
    """Utility class for advanced ShapeNet vector database operations."""
    
    def __init__(self, db: ShapeNetVectorDB):
        self.db = db
    
    def batch_similarity_search(
        self, 
        query_embeddings: List[np.ndarray], 
        limit: int = 10
    ) -> List[List[Dict[str, Any]]]:
        """
        Perform similarity search for multiple query embeddings.
        
        Args:
            query_embeddings: List of query embedding vectors
            limit: Number of results per query
            
        Returns:
            List of search results for each query
        """
        results = []
        for i, query_emb in enumerate(query_embeddings):
            print(f"Processing query {i+1}/{len(query_embeddings)}")
            query_results = self.db.search_similar(query_emb, limit=limit)
            results.append(query_results)
        return results
    
    def find_shape_by_pattern(self, pattern: str) -> List[str]:
        """
        Find shapes whose IDs match a pattern.
        
        Args:
            pattern: Pattern to match (supports * wildcard)
            
        Returns:
            List of matching shape IDs
        """
        import fnmatch
        shape_ids = self.db.list_shape_ids()
        return [sid for sid in shape_ids if fnmatch.fnmatch(sid, pattern)]
    
    def get_embedding_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the embeddings in the database.
        
        Returns:
            Dictionary with embedding statistics
        """
        if not self.db.is_indexed:
            return {"error": "Database not indexed"}
        
        # Sample a few embeddings to get statistics
        shape_ids = self.db.list_shape_ids(limit=100)
        embeddings = []
        
        for shape_id in shape_ids:
            shape_data = self.db.search_by_shape_id(shape_id)
            if shape_data and 'embedding' in shape_data:
                embeddings.append(shape_data['embedding'])
        
        if not embeddings:
            return {"error": "No embeddings found"}
        
        embeddings = np.array(embeddings)
        
        return {
            "num_samples": len(embeddings),
            "embedding_dim": embeddings.shape[1],
            "mean_norm": float(np.mean(np.linalg.norm(embeddings, axis=1))),
            "std_norm": float(np.std(np.linalg.norm(embeddings, axis=1))),
            "mean_values": embeddings.mean(axis=0).tolist()[:10],  # First 10 dimensions
            "std_values": embeddings.std(axis=0).tolist()[:10]     # First 10 dimensions
        }
    
    def export_search_results(
        self, 
        results: List[Dict[str, Any]], 
        output_file: str,
        include_embeddings: bool = False
    ):
        """
        Export search results to a JSON file.
        
        Args:
            results: Search results from similarity search
            output_file: Path to output JSON file
            include_embeddings: Whether to include embedding vectors in export
        """
        export_data = []
        
        for result in results:
            export_item = {
                "shape_id": result["shape_id"],
                "score": result.get("score"),
                "pc_path": result["pc_path"],
                "embedding_path": result["embedding_path"],
                "metadata": result.get("metadata", {})
            }
            
            if include_embeddings and "embedding" in result:
                # Convert numpy array to list for JSON serialization
                export_item["embedding"] = result["embedding"].tolist()
            
            export_data.append(export_item)
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"Exported {len(export_data)} results to {output_file}")
    
    def create_embedding_index_file(self, output_file: str = "embedding_index.json"):
        """
        Create an index file mapping shape IDs to file paths.
        
        Args:
            output_file: Path to output index file
        """
        if not self.db.is_indexed:
            raise ValueError("Database must be indexed first")
        
        index_data = {
            "total_shapes": len(self.db._shape_id_to_paths),
            "embedding_dim": 1280,  # Based on ULIP2
            "shapes": {}
        }
        
        for shape_id, (pc_path, emb_path) in self.db._shape_id_to_paths.items():
            index_data["shapes"][shape_id] = {
                "pc_path": pc_path,
                "embedding_path": emb_path
            }
        
        with open(output_file, 'w') as f:
            json.dump(index_data, f, indent=2)
        
        print(f"Created embedding index file: {output_file}")


def benchmark_search_performance(
    db: ShapeNetVectorDB,
    num_queries: int = 10,
    search_limits: List[int] = [1, 5, 10, 20]
) -> Dict[str, Any]:
    """
    Benchmark search performance of the database.
    
    Args:
        db: ShapeNet vector database
        num_queries: Number of random queries to test
        search_limits: Different limit values to test
        
    Returns:
        Dictionary with performance results
    """
    import time
    
    if not db.is_indexed:
        raise ValueError("Database must be indexed first")
    
    print(f"Benchmarking search performance with {num_queries} queries...")
    
    # Get random shape embeddings as queries
    shape_ids = db.list_shape_ids(limit=num_queries)
    query_embeddings = []
    
    for shape_id in shape_ids:
        shape_data = db.search_by_shape_id(shape_id)
        if shape_data and 'embedding' in shape_data:
            query_embeddings.append(shape_data['embedding'])
    
    results = {
        "num_queries": len(query_embeddings),
        "search_limits": search_limits,
        "performance": {}
    }
    
    for limit in search_limits:
        print(f"Testing with limit={limit}...")
        times = []
        
        for query_emb in query_embeddings:
            start_time = time.time()
            search_results = db.search_similar(query_emb, limit=limit)
            end_time = time.time()
            times.append(end_time - start_time)
        
        results["performance"][str(limit)] = {
            "avg_time": np.mean(times),
            "std_time": np.std(times),
            "min_time": np.min(times),
            "max_time": np.max(times)
        }
        
        print(f"  Average time: {np.mean(times):.4f}s")
    
    return results


def load_and_index_full_dataset(
    workspace: str = "./shapenet_vectordb_full",
    batch_size: int = 1000,
    pc_dir: str = "shapenet/shapenet_pc",
    embedding_dir: str = "shapenet/shapenet_embedding"
) -> ShapeNetVectorDB:
    """
    Load and index the full ShapeNet dataset.
    
    Args:
        workspace: Directory for the database
        batch_size: Batch size for indexing
        pc_dir: Point cloud directory
        embedding_dir: Embedding directory
        
    Returns:
        Indexed ShapeNet vector database
    """
    print("Creating vector database for full ShapeNet dataset...")
    
    db = create_shapenet_db(
        workspace=workspace,
        db_type="hnsw",  # Use HNSW for large datasets
        pc_dir=pc_dir,
        embedding_dir=embedding_dir,
        space='cosine',
        max_elements=100000,  # Adjust based on your dataset size
        ef_construction=400,   # Higher value for better recall
        ef=100,               # Higher value for better search quality
        M=32                  # Higher value for better connectivity
    )
    
    print("Starting full dataset indexing...")
    db.index_embeddings(batch_size=batch_size)
    
    print("Full dataset indexed successfully!")
    return db


def interactive_search_demo(db: ShapeNetVectorDB):
    """
    Interactive demo for exploring the vector database.
    
    Args:
        db: Indexed ShapeNet vector database
    """
    if not db.is_indexed:
        print("Database must be indexed first!")
        return
    
    print("\n" + "=" * 50)
    print("Interactive ShapeNet Vector Database Demo")
    print("=" * 50)
    print("Commands:")
    print("  'search <shape_id>' - Search for shapes similar to given shape ID")
    print("  'random' - Search with random embedding")
    print("  'stats' - Show database statistics")
    print("  'list' - List first 10 shape IDs")
    print("  'quit' - Exit demo")
    print()
    
    utils = ShapeNetDBUtils(db)
    
    while True:
        try:
            command = input("Enter command: ").strip().lower()
            
            if command == 'quit':
                break
            elif command == 'stats':
                stats = db.get_stats()
                emb_stats = utils.get_embedding_statistics()
                print("\nDatabase Statistics:")
                for key, value in stats.items():
                    print(f"  {key}: {value}")
                print("\nEmbedding Statistics:")
                for key, value in emb_stats.items():
                    if key not in ['mean_values', 'std_values']:
                        print(f"  {key}: {value}")
                        
            elif command == 'list':
                shape_ids = db.list_shape_ids(limit=10)
                print(f"\nFirst 10 Shape IDs:")
                for i, sid in enumerate(shape_ids, 1):
                    print(f"  {i}. {sid}")
                    
            elif command == 'random':
                print("\nSearching with random embedding...")
                random_emb = np.random.randn(1280).astype(np.float32)
                results = db.search_similar(random_emb, limit=5)
                print(f"Found {len(results)} results:")
                for i, result in enumerate(results, 1):
                    print(f"  {i}. {result['shape_id']} (score: {result['score']:.4f})")
                    
            elif command.startswith('search '):
                shape_id = command.split(' ', 1)[1]
                print(f"\nSearching for shapes similar to '{shape_id}'...")
                
                shape_data = db.search_by_shape_id(shape_id)
                if shape_data:
                    results = db.search_similar(shape_data['embedding'], limit=5)
                    print(f"Found {len(results)} results:")
                    for i, result in enumerate(results, 1):
                        print(f"  {i}. {result['shape_id']} (score: {result['score']:.4f})")
                else:
                    print(f"Shape '{shape_id}' not found in database")
                    
            else:
                print("Unknown command. Type 'quit' to exit.")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    # Example usage
    print("Creating database for testing...")
    db = create_shapenet_db()
    
    try:
        # Index a small subset for testing
        db.index_embeddings(batch_size=50, max_files=10)
        
        # Run interactive demo
        interactive_search_demo(db)
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure the shapenet directories exist and contain data.")
