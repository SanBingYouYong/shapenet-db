"""
Debug script to inspect the structure of search results from vectordb.
"""

import os
import numpy as np
from shapenet_vectordb import create_shapenet_db

def debug_search_results():
    """Debug the structure of search results."""
    print("Debugging VectorDB Search Results")
    print("=" * 40)
    
    # Create database
    db = create_shapenet_db(
        workspace="./debug_vectordb",
        db_type="hnsw",
        space='cosine',
        max_elements=100
    )
    
    # Index a few embeddings
    print("Indexing a few embeddings for debugging...")
    db.index_embeddings(batch_size=3, max_files=3)
    
    # Get a shape to use as query
    shape_ids = db.list_shape_ids(limit=1)
    if shape_ids:
        shape_data = db.search_by_shape_id(shape_ids[0])
        if shape_data:
            print(f"\nUsing shape '{shape_ids[0]}' as query")
            
            # Create query document
            from shapenet_vectordb import ShapeNetDoc
            from docarray import DocList
            
            query_embedding = shape_data['embedding']
            if query_embedding.ndim == 2 and query_embedding.shape[0] == 1:
                query_embedding = query_embedding.squeeze(0)
            
            query_doc = ShapeNetDoc(
                shape_id="query",
                embedding=query_embedding
            )
            
            # Perform raw search to inspect results
            print("\nPerforming raw search...")
            raw_results = db.db.search(inputs=DocList[ShapeNetDoc]([query_doc]), limit=3)
            
            print(f"Raw results type: {type(raw_results)}")
            print(f"Raw results length: {len(raw_results)}")
            
            if raw_results:
                first_result = raw_results[0]
                print(f"\nFirst result type: {type(first_result)}")
                print(f"First result attributes: {dir(first_result)}")
                
                if hasattr(first_result, 'matches'):
                    print(f"\nMatches type: {type(first_result.matches)}")
                    print(f"Number of matches: {len(first_result.matches)}")
                    
                    if first_result.matches:
                        first_match = first_result.matches[0]
                        print(f"\nFirst match type: {type(first_match)}")
                        print(f"First match attributes: {dir(first_match)}")
                        
                        # Check for score-related attributes
                        print(f"\nScore-related attributes:")
                        for attr in dir(first_match):
                            if 'score' in attr.lower():
                                print(f"  {attr}: {getattr(first_match, attr, 'N/A')}")
                        
                        # Check the main result object for scores
                        print(f"\nChecking main result object for scores:")
                        if hasattr(first_result, 'scores'):
                            print(f"  scores attribute: {first_result.scores}")
                            print(f"  scores type: {type(first_result.scores)}")
                            if first_result.scores:
                                print(f"  first score: {first_result.scores[0]}")
                        else:
                            print("  No scores attribute on main result")
                        
                        # Check each match individually for scores
                        print(f"\nChecking each match for score info:")
                        for i, match in enumerate(first_result.matches[:3]):
                            print(f"  Match {i}:")
                            print(f"    ID: {match.id}")
                            print(f"    Shape ID: {match.shape_id}")
                            if hasattr(match, 'score'):
                                print(f"    Score: {match.score}")
                            if hasattr(match, 'scores'):
                                print(f"    Scores: {match.scores}")
                        
                        # Check common attributes
                        common_attrs = ['id', 'shape_id', 'pc_path', 'embedding_path', 'metadata']
                        print(f"\nCommon attributes:")
                        for attr in common_attrs:
                            if hasattr(first_match, attr):
                                value = getattr(first_match, attr)
                                print(f"  {attr}: {value}")
                            else:
                                print(f"  {attr}: NOT FOUND")
                        
                        # Print all non-private attributes
                        print(f"\nAll non-private attributes:")
                        for attr in sorted(dir(first_match)):
                            if not attr.startswith('_'):
                                try:
                                    value = getattr(first_match, attr)
                                    if not callable(value):
                                        print(f"  {attr}: {type(value)} = {value}")
                                except:
                                    print(f"  {attr}: <error accessing>")

if __name__ == "__main__":
    debug_search_results()
