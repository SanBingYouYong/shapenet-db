"""
FastAPI server for ShapeNet text-based search with GLB conversion.
"""

import os
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
import uuid

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel

from shapenet_vectordb import ShapeNetVectorDB
from text_search import ShapeNetTextSearch
from retrieve_obj import retrieve_shapenet_model
from utils.obj2glb import obj_to_glb

# Initialize FastAPI app
app = FastAPI(
    title="ShapeNet Text Search API",
    description="REST API for text-based search and retrieval of ShapeNet models",
    version="1.0.0"
)

# Global variables for database and search engine
db: Optional[ShapeNetVectorDB] = None
text_search: Optional[ShapeNetTextSearch] = None

# Request/Response models
class SearchRequest(BaseModel):
    query: str
    limit: int = 1
    
class SearchResult(BaseModel):
    shape_id: str
    score: float
    query_text: str

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    total_found: int


@app.on_event("startup")
async def startup_event():
    """Initialize the vector database and text search on startup."""
    global db, text_search
    
    print("🚀 Starting ShapeNet Text Search API...")
    print("📊 Initializing vector database...")
    
    # Initialize vector database
    db = ShapeNetVectorDB()
    
    # Index embeddings if not already indexed
    if not db.is_indexed:
        print("⏳ Indexing embeddings (this may take a while)...")
        db.index_embeddings(batch_size=1000, max_files=None)
        print("✅ Vector database indexed successfully")
    else:
        print("✅ Vector database already indexed")
    
    # Initialize text search
    print("🔍 Initializing text search...")
    text_search = ShapeNetTextSearch(db)
    print("✅ Text search initialized successfully")
    print("🎉 API server ready!")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "ShapeNet Text Search API",
        "version": "1.0.0",
        "endpoints": {
            "search": "/search?query=<text>&limit=<number>",
            "search_and_download": "/search-glb?query=<text>&limit=<number>",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    global db, text_search
    
    status = "healthy" if (db is not None and text_search is not None) else "unhealthy"
    
    return {
        "status": status,
        "database_indexed": db.is_indexed if db else False,
        "text_search_ready": text_search is not None
    }


@app.get("/search", response_model=SearchResponse)
async def search_text(
    query: str = Query(..., description="Text query for searching similar shapes"),
    limit: int = Query(1, ge=1, le=50, description="Maximum number of results to return")
):
    """
    Search for ShapeNet models using text query.
    
    Args:
        query: Text description of the desired shape
        limit: Number of results to return (1-50)
    
    Returns:
        Search results with shape IDs and similarity scores
    """
    global text_search
    
    if text_search is None:
        raise HTTPException(status_code=503, detail="Text search not initialized")
    
    try:
        # Perform text search
        results = text_search.search_by_text(query, limit=limit)
        
        # Convert results to response format
        search_results = [
            SearchResult(
                shape_id=result['shape_id'],
                score=result['score'] if result['score'] is not None else 0.0,
                query_text=result['query_text']
            )
            for result in results
        ]
        
        return SearchResponse(
            query=query,
            results=search_results,
            total_found=len(search_results)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.get("/search-glb")
async def search_and_download_glb(
    query: str = Query(..., description="Text query for searching similar shapes"),
    limit: int = Query(1, ge=1, le=10, description="Maximum number of results to return"),
    output_dir: str = Query("./retrieved", description="Directory to save retrieved models")
):
    """
    Search for ShapeNet models using text query and return the best match as GLB file.
    
    Args:
        query: Text description of the desired shape
        limit: Number of results to consider (will return the best match)
        output_dir: Directory to save retrieved models
    
    Returns:
        GLB file of the best matching model
    """
    global text_search
    
    if text_search is None:
        raise HTTPException(status_code=503, detail="Text search not initialized")
    
    try:
        # Perform text search
        results = text_search.search_by_text(query, limit=limit)
        
        if not results:
            raise HTTPException(status_code=404, detail=f"No results found for query: {query}")
        
        # Get the best result
        best_result = results[0]
        shape_id = best_result['shape_id']
        
        print(f"📥 Retrieving model for shape_id: {shape_id}")
        
        # Retrieve the ShapeNet model
        retrieve_shapenet_model(
            shape_id=shape_id,
            output_dir=output_dir,
            zip_root_dir="hf_shapenet_zips"
        )
        
        # Find the OBJ file
        synset_id, object_id = shape_id.split("-")
        obj_path = Path(output_dir) / synset_id / object_id / "models" / "model_normalized.obj"
        
        if not obj_path.exists():
            raise HTTPException(status_code=404, detail=f"OBJ file not found: {obj_path}")
        
        # Create temporary GLB file
        temp_dir = Path(tempfile.gettempdir())
        glb_filename = f"shapenet_{shape_id}_{uuid.uuid4().hex[:8]}.glb"
        glb_path = temp_dir / glb_filename
        
        print(f"🔄 Converting OBJ to GLB: {obj_path} -> {glb_path}")
        
        # Convert OBJ to GLB
        obj_to_glb(str(obj_path.absolute()), str(glb_path.absolute()))
        
        if not glb_path.exists():
            raise HTTPException(status_code=500, detail="GLB conversion failed")
        
        print(f"✅ GLB conversion completed: {glb_path}")
        
        # Return the GLB file
        return FileResponse(
            path=str(glb_path),
            filename=glb_filename,
            media_type="model/gltf-binary",
            headers={
                "Content-Disposition": f"attachment; filename={glb_filename}",
                "X-Shape-ID": shape_id,
                "X-Query": query,
                "X-Score": str(best_result['score']) if best_result['score'] is not None else "N/A"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search and conversion failed: {str(e)}")


@app.post("/search", response_model=SearchResponse)
async def search_text_post(request: SearchRequest):
    """
    Search for ShapeNet models using text query (POST version).
    
    Args:
        request: Search request with query and limit
    
    Returns:
        Search results with shape IDs and similarity scores
    """
    return await search_text(query=request.query, limit=request.limit)


if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting ShapeNet Text Search API server...")
    print("📍 Server will be available at: http://localhost:8001")
    print("📖 API documentation at: http://localhost:8001/docs")
    
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8001,
        reload=False,  # Set to True for development
        log_level="info"
    )
