# ShapeNet Vector Database

A Python vector database for ShapeNet embeddings with similarity search and point cloud mapping capabilities, built on top of [Jina AI's VectorDB](https://github.com/jina-ai/vectordb).

## Features

- **Fast Similarity Search**: Uses HNSW (Hierarchical Navigable Small World) algorithm for efficient approximate nearest neighbor search
- **Text-Based Search**: Search shapes using natural language descriptions via CLIP model
- **Point Cloud Mapping**: Maps search results back to original ShapeNet point clouds
- **Batch Processing**: Efficient batch indexing and searching capabilities
- **Flexible Database Types**: Support for both exact and approximate search
- **Export Capabilities**: Export search results and create embedding indexes
- **Performance Benchmarking**: Built-in tools for performance analysis

## Directory Structure

Your project should have the following structure:
```
ShapenetDB/
├── shapenet/
│   ├── shapenet_pc/          # 50k+ .npy point cloud files
│   └── shapenet_embedding/   # 50k+ .npy embedding files (with _embedding suffix)
├── models/
│   └── open_clip_pytorch_model.bin  # CLIP model for text search
├── shapenet_vectordb.py      # Main vector database implementation
├── text_search.py            # Text search functionality using CLIP
├── shapenet_utils.py         # Utility functions
├── examples.py               # Usage examples
├── text_search_examples.py   # Text search examples
├── main.py                   # Demo script
└── README.md                 # This file
```

## Quick Start

### 0. Dependencies

1. `uv sync`
2. to use newer torch versions, overwrite open-clip-torch's torch, e.g.: `uv pip install --upgrade --force-reinstall torch --index-url https://download.pytorch.org/whl/cu128`. 

## Quick Start

### 1. Basic Usage

```python
from shapenet_vectordb import create_shapenet_db

# Create the vector database
db = create_shapenet_db(
    workspace="./my_vectordb",
    db_type="hnsw",  # or "exact" for exact search
    pc_dir="shapenet/shapenet_pc",
    embedding_dir="shapenet/shapenet_embedding"
)

# Index embeddings (start with a subset for testing)
db.index_embeddings(batch_size=100, max_files=1000)

# Search for similar shapes
shape_ids = db.list_shape_ids(limit=1)
shape_data = db.search_by_shape_id(shape_ids[0])
similar_shapes = db.search_similar(
    shape_data['embedding'], 
    limit=10,
    return_point_clouds=True
)

# Access results
for result in similar_shapes:
    print(f"Shape: {result['shape_id']}, Score: {result['score']:.4f}")
    if 'point_cloud' in result:
        print(f"Point cloud shape: {result['point_cloud'].shape}")
```

### 2. Text-Based Search

```python
from text_search import create_text_search

# Create text search (requires CLIP model)
text_search = create_text_search(db)

# Search using natural language
results = text_search.search_by_text("wooden chair with four legs", limit=5)

# Access results
for result in results:
    print(f"Shape: {result['shape_id']}, Score: {result['score']:.4f}")
    print(f"Query: {result['query_text']}")
```

### 3. Batch Text Search

```python
# Search multiple queries at once
queries = ["chair", "table", "car", "airplane"]
batch_results = text_search.batch_text_search(queries, limit=3)

for query, results in zip(queries, batch_results):
    print(f"Results for '{query}':")
    for result in results:
        print(f"  {result['shape_id']} (score: {result['score']:.4f})")
```

### 2. Text-Based Search

```python
from text_search import create_text_search

# Create text search (requires CLIP model)
text_search = create_text_search(db)

# Search using natural language
results = text_search.search_by_text("wooden chair with four legs", limit=5)

# Access results
for result in results:
    print(f"Shape: {result['shape_id']}, Score: {result['score']:.4f}")
    print(f"Query: {result['query_text']}")
```

### 3. Custom Embedding Query

```python
import numpy as np

# Create a custom query embedding (e.g., from your model)
custom_embedding = np.random.randn(1280).astype(np.float32)

# Search with custom embedding
results = db.search_similar(custom_embedding, limit=5)
```

### 4. Custom Embedding Query

```python
import numpy as np

# Create a custom query embedding (e.g., from your model)
custom_embedding = np.random.randn(1280).astype(np.float32)

# Search with custom embedding
results = db.search_similar(custom_embedding, limit=5)
```

### 5. Shape ID Lookup

```python
# Find a specific shape by ID
shape_data = db.search_by_shape_id("your_shape_id")
if shape_data:
    point_cloud = shape_data['point_cloud']
    embedding = shape_data['embedding']
```

## Running the Examples

### Demo Script
```bash
python main.py
```

### Comprehensive Examples
```bash
python examples.py
```

### Text Search Examples
```bash
python text_search_examples.py
```

### Test Text Search
```bash
python test_text_search.py
```

### Final Comprehensive Demo
```bash
python final_demo.py
```

### Interactive Mode
```bash
python final_demo.py --interactive
```

## Configuration Options

### Database Types

- **HNSW (Recommended for large datasets)**:
  ```python
  db = create_shapenet_db(
      db_type="hnsw",
      space='cosine',        # 'cosine', 'l2', or 'ip'
      max_elements=100000,   # Maximum number of elements
      ef_construction=200,   # Higher = better recall, slower indexing
      ef=50,                 # Higher = better search quality, slower search
      M=16                   # Higher = better connectivity, more memory
  )
  ```

- **Exact Search (Smaller datasets)**:
  ```python
  db = create_shapenet_db(db_type="exact")
  ```

### Indexing Options

```python
# Index all embeddings
db.index_embeddings(batch_size=1000)

# Index subset for testing
db.index_embeddings(batch_size=100, max_files=1000)
```

## API Reference

### ShapeNetVectorDB

#### Methods

- `index_embeddings(batch_size=1000, max_files=None)`: Index embeddings from files
- `search_similar(query_embedding, limit=10, return_point_clouds=False)`: Search for similar shapes
- `search_by_shape_id(shape_id)`: Get data for specific shape ID
- `get_stats()`: Get database statistics
- `list_shape_ids(limit=None)`: List all indexed shape IDs

#### Search Results Format

```python
{
    'shape_id': 'shape_identifier',
    'score': 0.95,  # Similarity score
    'pc_path': '/path/to/pointcloud.npy',
    'embedding_path': '/path/to/embedding.npy',
    'metadata': {...},
    'point_cloud': np.array(...),  # If return_point_clouds=True
    'embedding': np.array(...)     # If available
}
```

### ShapeNetDBUtils

Utility class for advanced operations:

- `batch_similarity_search(query_embeddings, limit=10)`: Batch search
- `find_shape_by_pattern(pattern)`: Find shapes matching pattern
- `get_embedding_statistics()`: Get embedding statistics
- `export_search_results(results, output_file)`: Export results to JSON
- `create_embedding_index_file(output_file)`: Create index file

### ShapeNetTextSearchUtils

Utility class for text search operations:

- `batch_text_search_with_export(queries, output_dir)`: Batch text search with export
- `evaluate_text_search_quality(test_queries)`: Evaluate text search performance

## Performance Optimization

### For Large Datasets (50k+ shapes)

1. **Use HNSW database**:
   ```python
   db = create_shapenet_db(
       db_type="hnsw",
       space='cosine',
       max_elements=100000,
       ef_construction=400,  # Higher for better recall
       ef=100,               # Higher for better search quality
       M=32                  # Higher for better connectivity
   )
   ```

2. **Optimize batch size**:
   ```python
   db.index_embeddings(batch_size=2000)  # Larger batches for efficiency
   ```

3. **Monitor memory usage**:
   - HNSW uses more memory but provides faster search
   - Adjust `max_elements` based on your dataset size

### For Real-time Applications

- Use HNSW with moderate `ef` values (50-100)
- Pre-index all embeddings
- Consider using multiple database instances for different shape categories

## Data Format Requirements

### Point Clouds
- Format: `.npy` files
- Shape: Variable (typically N x 3 for N points)
- Location: `shapenet/shapenet_pc/`
- Naming: `{shape_id}.npy`

### Embeddings
- Format: `.npy` files  
- Shape: `(1, 1280)` or `(1280,)` - ULIP2 embedding dimension
- Location: `shapenet/shapenet_embedding/`
- Naming: `{shape_id}_embedding.npy`

## Troubleshooting

### Common Issues

1. **"No matching files found"**:
   - Check that point cloud and embedding files have corresponding names
   - Verify directory paths are correct

2. **Memory errors during indexing**:
   - Reduce `batch_size` parameter
   - Use HNSW instead of exact search for large datasets

3. **Slow search performance**:
   - Increase `ef` parameter for HNSW
   - Ensure database is properly indexed

4. **Import errors**:
   - Install dependencies: `pip install vectordb numpy`
   - Check Python version (requires 3.11+)

### Debugging

Enable verbose output:
```python
import logging
logging.basicConfig(level=logging.INFO)
```

## Dependencies

- Python >= 3.11
- vectordb >= 0.0.21
- open-clip-torch >= 2.24.0
- torch (automatically installed with open-clip-torch)
- numpy
- docarray (automatically installed with vectordb)

### Installing Dependencies

```bash
pip install vectordb open-clip-torch numpy
```

### CLIP Model Setup

The text search functionality requires a pre-trained CLIP model. Place the model file at:
```
models/open_clip_pytorch_model.bin
```

The system uses the ViT-bigG-14 CLIP model architecture compatible with ULIP2 embeddings.

## License

This project follows the same license as the underlying VectorDB library (Apache-2.0).

## Contributing

Contributions are welcome! Please ensure your code follows the existing style and includes appropriate tests.
