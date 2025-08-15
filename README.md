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
│   └── shapenet_embedding/   # 50k+ .npy embedding files (with _embedding suffix) produced by ulip2_encoder
├── hf_shapenet_zips/         # Downloaded ShapeNetCore.v2 (gated) dataset from Huggingface, the path to snapshot/hash/ folder containing ZIP files for un-extracted OBJ file retrieval, each zip named after synset id.
├── models/
│   └── open_clip_pytorch_model.bin  # CLIP model for text search, used by ULIP2
├── shapenet_vectordb.py      # Vector database for ShapeNet point cloud and embeddingsimplementation
├── text_search.py            # Text search functionality using CLIP
├── main.py                   # Main entrypoint
└── README.md                 # This file
```

## Quick Start

### 0. Dependencies

1. `uv sync` or `pip install -r requirements.txt`
2. (optional) to use newer torch versions, overwrite open-clip-torch's torch, e.g.: `uv pip install --upgrade --force-reinstall torch --index-url https://download.pytorch.org/whl/cu128`. 
3. Download open-clip ViT model weights to use with ulip2-encoded shapenet embeddings: `bash download_model.sh`
4. Get access and download ShapeNetCore.v2 dataset. 
5. Use [ulip2 encoder](https://github.com/SanBingYouYong/ulip2_encoder) to embed all shapenet point cloud (using the point cloud from ulip's triplets)
    - the coverage of ULIP shapenet triplet on shapenet core v2 is tested

### 0.1 Running in Docker

0. build the image with `docker build -t shapenet_db .`
1. run command: replace any symlinks to actual paths

```bash
docker run -it --rm --gpus all  -v $(pwd)/models:/app/models   -v $(pwd)/shapenet:/app/shapenet   -v $(pwd)/hf_shapenet_zips:/app/hf_shapenet_zips   -v $(pwd)/retrieved:/app/retrieved   -v $(pwd)/shapenet_vectordb:/app/shapenet_vectordb   shapenet_db:latest
```

e.g.: (in addition, we need to mount blob/hash files accroding to zip (actually symlinks))

```bash
docker run -it --gpus all -v /home/shuyuan/ULIP/ulip_models/:/app/models -v /home/shuyuan/ULIP/shapenet/:/app/shapenet -v /home/shuyuan/.cache/huggingface/hub/datasets--ShapeNet--ShapeNetCore/snapshots/0efb24cbe6828a85771a28335c5f7b5626514d9b:/app/hf_shapenet_zips -v /home/shuyuan/.cache/huggingface/hub/datasets--ShapeNet--ShapeNetCore/blobs:/blobs -v $(pwd)/retrieved:/app/retrieved -v $(pwd)/shapenet_vectordb:/app/shapenet_vectordb -p 8001:8001 --name shapenet_db shapenet_db_server:latest
```
- please remember to clean up ./retrieved dir in case many files are retrieved

### 1. Basic Usage

`python main.py --text <query, e.g. chair>` will retrieve a best match 3D model and put the content under `retrieved/` folder. 
- for more usage examples, check args in main.py

## License

This project follows the same license as the underlying VectorDB library (Apache-2.0).
- ULIP2 uses BSD-3, the CLIP model involved uses MIT. ShapeNet is a gated resource. 

## Contributing

Contributions are welcome! Please ensure your code follows the existing style and includes appropriate tests.
- this can apply to other 3D dataset too, theoretically.
