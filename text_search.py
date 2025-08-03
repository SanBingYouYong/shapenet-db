"""
Text-based search functionality for ShapeNet Vector Database using CLIP.
"""

import os
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Union
import open_clip
from pathlib import Path

from shapenet_vectordb import ShapeNetVectorDB


class ShapeNetTextSearch:
    """
    Text-based search for ShapeNet vector database using CLIP model.
    """
    
    def __init__(
        self, 
        vectordb: ShapeNetVectorDB,
        model_path: str = "models/open_clip_pytorch_model.bin",
        model_name: str = "ViT-bigG-14",
        device: Optional[str] = None
    ):
        """
        Initialize text search with CLIP model.
        
        Args:
            vectordb: ShapeNet vector database instance
            model_path: Path to the CLIP model checkpoint
            model_name: Name of the CLIP model architecture
            device: Device to run the model on ('cuda', 'cpu', or None for auto)
        """
        self.vectordb = vectordb
        self.model_path = model_path
        self.model_name = model_name
        
        # Set device
        if device is None:
            # Try CUDA first, but fall back to CPU if there are issues
            if torch.cuda.is_available():
                try:
                    # Test CUDA with a simple operation
                    test_tensor = torch.tensor([1.0]).cuda()
                    _ = test_tensor * 2  # Simple operation to test CUDA
                    self.device = "cuda"
                except Exception as e:
                    print(f"CUDA available but not working properly: {e}")
                    print("Falling back to CPU...")
                    self.device = "cpu"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        
        # Load CLIP model
        self._load_model()
    
    def _load_model(self):
        """Load the CLIP model from checkpoint."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"CLIP model not found at: {self.model_path}")
        
        print(f"Loading CLIP model from: {self.model_path}")
        
        try:
            # Load the model from local checkpoint
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                self.model_name, 
                pretrained=self.model_path
            )
            
            # Move model to device and set to eval mode
            self.model = self.model.to(self.device)
            self.model.eval()
            
            print(f"✅ CLIP model loaded successfully on {self.device}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load CLIP model: {e}")
    
    def encode_text(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Encode text using CLIP model.
        
        Args:
            text: Text string or list of text strings to encode
            
        Returns:
            Text embedding(s) as numpy array
        """
        if isinstance(text, str):
            text = [text]
        
        try:
            with torch.no_grad():
                # Tokenize text
                text_tokens = open_clip.tokenize(text).to(self.device)
                
                # Encode text
                text_features = self.model.encode_text(text_tokens)
                
                # Normalize features (important for similarity search)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                # Convert to numpy
                embeddings = text_features.cpu().numpy()
                
                # Return single embedding if input was single string
                if len(text) == 1:
                    return embeddings[0]
                
                return embeddings
                
        except Exception as e:
            raise RuntimeError(f"Failed to encode text: {e}")
    
    def search_by_text(
        self, 
        text_query: str, 
        limit: int = 10,
        return_point_clouds: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Search for shapes using text description.
        
        Args:
            text_query: Text description of the desired shape
            limit: Number of results to return
            return_point_clouds: Whether to load point cloud data
            
        Returns:
            List of search results with similarity scores
        """
        if not self.vectordb.is_indexed:
            raise ValueError("Vector database is not indexed yet!")
        
        print(f"Searching for: '{text_query}'")
        
        # Encode text query
        text_embedding = self.encode_text(text_query)
        
        # Search in vector database
        results = self.vectordb.search_similar(
            query_embedding=text_embedding,
            limit=limit,
            return_point_clouds=return_point_clouds
        )
        
        # Add query text to results for reference
        for result in results:
            result['query_text'] = text_query
        
        return results
    
    def batch_text_search(
        self, 
        text_queries: List[str], 
        limit: int = 10,
        return_point_clouds: bool = False
    ) -> List[List[Dict[str, Any]]]:
        """
        Search for multiple text queries in batch.
        
        Args:
            text_queries: List of text descriptions
            limit: Number of results per query
            return_point_clouds: Whether to load point cloud data
            
        Returns:
            List of search results for each query
        """
        print(f"Batch searching for {len(text_queries)} text queries...")
        
        all_results = []
        for i, query in enumerate(text_queries):
            print(f"Processing query {i+1}/{len(text_queries)}: '{query}'")
            results = self.search_by_text(
                query, 
                limit=limit, 
                return_point_clouds=return_point_clouds
            )
            all_results.append(results)
        
        return all_results
    
    def compare_text_embeddings(
        self, 
        text1: str, 
        text2: str
    ) -> float:
        """
        Compare similarity between two text descriptions.
        
        Args:
            text1: First text description
            text2: Second text description
            
        Returns:
            Cosine similarity score between the texts
        """
        emb1 = self.encode_text(text1)
        emb2 = self.encode_text(text2)
        
        # Cosine similarity
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        
        return float(similarity)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded CLIP model."""
        return {
            'model_name': self.model_name,
            'model_path': self.model_path,
            'device': self.device,
            'embedding_dim': self.model.text_projection.shape[1] if hasattr(self.model, 'text_projection') else 'unknown'
        }


def create_text_search(
    vectordb: ShapeNetVectorDB,
    model_path: str = "models/open_clip_pytorch_model.bin",
    model_name: str = "ViT-bigG-14",
    device: Optional[str] = None
) -> ShapeNetTextSearch:
    """
    Convenience function to create a text search instance.
    
    Args:
        vectordb: ShapeNet vector database instance
        model_path: Path to the CLIP model checkpoint
        model_name: Name of the CLIP model architecture
        device: Device to run the model on
        
    Returns:
        ShapeNetTextSearch instance
    """
    return ShapeNetTextSearch(
        vectordb=vectordb,
        model_path=model_path,
        model_name=model_name,
        device=device
    )


# Predefined text queries for common shapes
COMMON_SHAPE_QUERIES = [
    # Furniture
    "chair with four legs",
    "wooden table",
    "office chair with wheels",
    "round dining table",
    "bookshelf with multiple shelves",
    "sofa with cushions",
    "bed with headboard",
    "desk lamp",
    
    # Vehicles
    "car with four wheels",
    "airplane with wings",
    "motorcycle",
    "bicycle",
    "truck",
    "helicopter with rotor blades",
    
    # Electronics
    "computer monitor",
    "laptop computer",
    "telephone",
    "radio",
    "television",
    
    # Kitchen items
    "coffee mug",
    "wine bottle",
    "kitchen knife",
    "plate",
    "bowl",
    
    # Tools and objects
    "screwdriver",
    "hammer",
    "scissors",
    "pen",
    "book",
    "clock",
    
    # Animals (if present in dataset)
    "dog",
    "cat",
    "bird",
    "fish",
    
    # Abstract descriptions
    "round object",
    "long thin object",
    "square shaped item",
    "curved surface",
    "geometric shape",
    "symmetrical object"
]


def demo_text_search_queries() -> List[str]:
    """Get a list of demo text queries for testing."""
    return [
        "wooden chair",
        "round table",
        "car with wheels",
        "airplane",
        "computer laptop",
        "coffee cup",
        "book",
        "lamp with shade"
    ]
