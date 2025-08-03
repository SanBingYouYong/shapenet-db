"""
ShapeNet Vector Database

A vector database for ShapeNet embeddings with similarity search and point cloud mapping.
"""

import os
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass

from docarray import BaseDoc, DocList
from docarray.typing import NdArray
from vectordb import InMemoryExactNNVectorDB, HNSWVectorDB


class ShapeNetDoc(BaseDoc):
    """Document schema for ShapeNet embeddings."""
    
    # Unique identifier for the shape (filename without extension)
    shape_id: str = ''
    
    # Path to the point cloud file
    pc_path: str = ''
    
    # Path to the embedding file
    embedding_path: str = ''
    
    # The actual embedding vector (1280-dimensional based on ULIP2)
    embedding: NdArray[1280]
    
    # Optional metadata
    category: Optional[str] = None
    
    # Additional metadata that can be stored
    metadata: Dict[str, Any] = {}


class ShapeNetVectorDB:
    """
    Vector database for ShapeNet embeddings with similarity search capabilities.
    """
    
    def __init__(
        self, 
        workspace: str = "./shapenet_vectordb",
        db_type: str = "hnsw",
        pc_dir: str = "shapenet/shapenet_pc",
        embedding_dir: str = "shapenet/shapenet_embedding",
        **db_kwargs
    ):
        """
        Initialize the ShapeNet Vector Database.
        
        Args:
            workspace: Directory to store the vector database
            db_type: Type of vector database ("hnsw" or "exact")
            pc_dir: Directory containing point cloud .npy files
            embedding_dir: Directory containing embedding .npy files
            **db_kwargs: Additional arguments for the vector database
        """
        self.workspace = workspace
        self.pc_dir = pc_dir
        self.embedding_dir = embedding_dir
        
        # Create workspace directory if it doesn't exist
        os.makedirs(workspace, exist_ok=True)
        
        # Initialize the vector database
        if db_type.lower() == "hnsw":
            # HNSW for approximate nearest neighbor search (faster, good for large datasets)
            default_hnsw_params = {
                'space': 'cosine',  # Use cosine similarity for embeddings
                'max_elements': 100000,  # Allow up to 100k elements (adjust as needed)
                'ef_construction': 200,
                'ef': 50,
                'M': 16
            }
            default_hnsw_params.update(db_kwargs)
            self.db = HNSWVectorDB[ShapeNetDoc](workspace=workspace, **default_hnsw_params)
        else:
            # Exact nearest neighbor search (slower but exact results)
            self.db = InMemoryExactNNVectorDB[ShapeNetDoc](workspace=workspace)
        
        self.is_indexed = False
        self._shape_id_to_paths = {}
    
    def _get_shape_id_from_filename(self, filename: str) -> str:
        """Extract shape ID from filename (remove extension and _embedding suffix)."""
        shape_id = filename.replace('.npy', '')
        if shape_id.endswith('_embedding'):
            shape_id = shape_id[:-10]  # Remove '_embedding' suffix
        return shape_id
    
    def _find_corresponding_files(self) -> Dict[str, Tuple[str, str]]:
        """
        Find corresponding point cloud and embedding files.
        
        Returns:
            Dictionary mapping shape_id to (pc_path, embedding_path)
        """
        shape_files = {}
        
        # Get all embedding files
        if not os.path.exists(self.embedding_dir):
            raise FileNotFoundError(f"Embedding directory not found: {self.embedding_dir}")
            
        embedding_files = [f for f in os.listdir(self.embedding_dir) if f.endswith('.npy')]
        
        for emb_file in embedding_files:
            shape_id = self._get_shape_id_from_filename(emb_file)
            
            # Find corresponding point cloud file
            pc_file = f"{shape_id}.npy"
            pc_path = os.path.join(self.pc_dir, pc_file)
            embedding_path = os.path.join(self.embedding_dir, emb_file)
            
            # Check if point cloud file exists
            if os.path.exists(pc_path):
                shape_files[shape_id] = (pc_path, embedding_path)
            else:
                print(f"Warning: Point cloud file not found for {shape_id}: {pc_path}")
        
        return shape_files
    
    def index_embeddings(self, batch_size: int = 1000, max_files: Optional[int] = None):
        """
        Index all embeddings from the embedding directory.
        
        Args:
            batch_size: Number of embeddings to process in each batch
            max_files: Maximum number of files to process (for testing)
        """
        print("Finding corresponding point cloud and embedding files...")
        shape_files = self._find_corresponding_files()
        
        if not shape_files:
            raise ValueError("No matching point cloud and embedding files found!")
        
        print(f"Found {len(shape_files)} matching shape files")
        
        # Limit files if max_files is specified
        if max_files:
            shape_files = dict(list(shape_files.items())[:max_files])
            print(f"Limited to {len(shape_files)} files for processing")
        
        # Store the mapping for later use
        self._shape_id_to_paths = shape_files
        
        # Process in batches
        shape_ids = list(shape_files.keys())
        total_batches = (len(shape_ids) + batch_size - 1) // batch_size
        
        print(f"Processing {len(shape_ids)} embeddings in {total_batches} batches...")
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(shape_ids))
            batch_shape_ids = shape_ids[start_idx:end_idx]
            
            print(f"Processing batch {batch_idx + 1}/{total_batches} ({len(batch_shape_ids)} items)...")
            
            # Create documents for this batch
            docs = []
            for shape_id in batch_shape_ids:
                pc_path, embedding_path = shape_files[shape_id]
                
                try:
                    # Load embedding
                    embedding = np.load(embedding_path)
                    
                    # Handle different embedding shapes
                    if embedding.ndim == 2 and embedding.shape[0] == 1:
                        embedding = embedding.squeeze(0)  # Remove batch dimension
                    elif embedding.ndim != 1:
                        print(f"Warning: Unexpected embedding shape for {shape_id}: {embedding.shape}")
                        continue
                    
                    # Create document
                    doc = ShapeNetDoc(
                        shape_id=shape_id,
                        pc_path=pc_path,
                        embedding_path=embedding_path,
                        embedding=embedding,
                        metadata={'indexed_at': 'batch_' + str(batch_idx)}
                    )
                    docs.append(doc)
                    
                except Exception as e:
                    print(f"Error processing {shape_id}: {e}")
                    continue
            
            if docs:
                # Index this batch
                doc_list = DocList[ShapeNetDoc](docs)
                self.db.index(inputs=doc_list)
                print(f"Indexed batch {batch_idx + 1} with {len(docs)} documents")
            else:
                print(f"No valid documents in batch {batch_idx + 1}")
        
        self.is_indexed = True
        print(f"Indexing complete! Total documents indexed: {len(self._shape_id_to_paths)}")
    
    def search_similar(
        self, 
        query_embedding: np.ndarray, 
        limit: int = 10,
        return_point_clouds: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Search for similar embeddings.
        
        Args:
            query_embedding: Query embedding vector
            limit: Number of similar results to return
            return_point_clouds: Whether to load and return point cloud data
            
        Returns:
            List of dictionaries containing search results
        """
        if not self.is_indexed:
            raise ValueError("Database not indexed yet! Call index_embeddings() first.")
        
        # Handle different query embedding shapes
        if query_embedding.ndim == 2 and query_embedding.shape[0] == 1:
            query_embedding = query_embedding.squeeze(0)
        
        # Create query document
        query_doc = ShapeNetDoc(
            shape_id="query",
            embedding=query_embedding
        )
        
        # Perform search
        results = self.db.search(inputs=DocList[ShapeNetDoc]([query_doc]), limit=limit)
        
        # Process results
        search_results = []
        
        # Get the scores from the main result object
        scores = results[0].scores if hasattr(results[0], 'scores') and results[0].scores else []
        
        for i, match in enumerate(results[0].matches):
            # Get the corresponding score for this match
            score = float(scores[i]) if i < len(scores) else None
            
            result = {
                'shape_id': match.shape_id,
                'score': score,
                'pc_path': match.pc_path,
                'embedding_path': match.embedding_path,
                'metadata': match.metadata
            }
            
            # Optionally load point cloud data
            if return_point_clouds and os.path.exists(match.pc_path):
                try:
                    result['point_cloud'] = np.load(match.pc_path)
                except Exception as e:
                    print(f"Error loading point cloud for {match.shape_id}: {e}")
                    result['point_cloud'] = None
            
            search_results.append(result)
        
        return search_results
    
    def search_by_shape_id(self, shape_id: str) -> Optional[Dict[str, Any]]:
        """
        Search for a specific shape by its ID.
        
        Args:
            shape_id: The shape ID to search for
            
        Returns:
            Dictionary containing shape information or None if not found
        """
        if shape_id in self._shape_id_to_paths:
            pc_path, embedding_path = self._shape_id_to_paths[shape_id]
            
            try:
                point_cloud = np.load(pc_path)
                embedding = np.load(embedding_path)
                
                return {
                    'shape_id': shape_id,
                    'pc_path': pc_path,
                    'embedding_path': embedding_path,
                    'point_cloud': point_cloud,
                    'embedding': embedding
                }
            except Exception as e:
                print(f"Error loading data for {shape_id}: {e}")
                return None
        else:
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        return {
            'total_indexed': len(self._shape_id_to_paths),
            'is_indexed': self.is_indexed,
            'workspace': self.workspace,
            'pc_dir': self.pc_dir,
            'embedding_dir': self.embedding_dir
        }
    
    def list_shape_ids(self, limit: Optional[int] = None) -> List[str]:
        """List all indexed shape IDs."""
        shape_ids = list(self._shape_id_to_paths.keys())
        if limit:
            return shape_ids[:limit]
        return shape_ids


def create_shapenet_db(
    workspace: str = "./shapenet_vectordb",
    db_type: str = "hnsw",
    pc_dir: str = "shapenet/shapenet_pc", 
    embedding_dir: str = "shapenet/shapenet_embedding",
    **kwargs
) -> ShapeNetVectorDB:
    """
    Convenience function to create a ShapeNet vector database.
    
    Args:
        workspace: Directory to store the vector database
        db_type: Type of vector database ("hnsw" or "exact")
        pc_dir: Directory containing point cloud .npy files
        embedding_dir: Directory containing embedding .npy files
        **kwargs: Additional arguments for the vector database
        
    Returns:
        ShapeNetVectorDB instance
    """
    return ShapeNetVectorDB(
        workspace=workspace,
        db_type=db_type, 
        pc_dir=pc_dir,
        embedding_dir=embedding_dir,
        **kwargs
    )
