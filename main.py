from shapenet_vectordb import ShapeNetVectorDB
from text_search import ShapeNetTextSearch
from retrieve_obj import retrieve_shapenet_model

import argparse

def main():
    # args
    parser = argparse.ArgumentParser(description="ShapeNet Retrieval Based On Point Cloud and Text Similarity Search")
    parser.add_argument(
        "--index_limit", type=int, default=None, 
        help="Limit the number of shapes to index (default: all)"
    )
    parser.add_argument(
        "--index_batch_size", type=int, default=1000,
        help="Batch size for indexing shapes (default: 1000)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./retrieved",
        help="Directory to save retrieved models (default: ./retrieved_shapenet_models)"
    )
    parser.add_argument(
        "--search_limit", type=int, default=1,
        help="Limit the number of search results to return (default: 1)"
    )
    parser.add_argument(
        "--zip_path", type=str, default="hf_shapenet_zips",
        help="Path to the ShapeNet ZIP files (default: hf_shapenet_zips)"
    )
    # text query
    parser.add_argument(
        "--text", type=str, 
        help="Text query for searching similar shapes (default: 'chair')"
    )
    args = parser.parse_args()
    if not args.text:
        raise ValueError("Text query must be provided!")
    print(f"Indexing limit: {args.index_limit}, Batch size: {args.index_batch_size}")
    print(f"Output directory: {args.output_dir}")
    print(f"Text query: {args.text}")
    # create vector database
    db = ShapeNetVectorDB()
    db.index_embeddings(
        batch_size=args.index_batch_size,
        max_files=args.index_limit
    )
    print("Vector database indexed successfully.")
    # create text search
    text_search = ShapeNetTextSearch(db)
    print("Text search initialized successfully.")
    # search by text
    results = text_search.search_by_text(args.text, limit=args.search_limit)
    if results:
        print(f"Found {len(results)} results for query '{args.text}':")
        for i, result in enumerate(results, 1):
            score_str = f"{result['score']:.4f}" if result['score'] is not None else "N/A"
            print(f"  {i}. {result['shape_id']} (score: {score_str})")
            # retrieve and save model
            retrieve_shapenet_model(
                shape_id=result['shape_id'],
                output_dir=args.output_dir,
                zip_root_dir=args.zip_path
            )
    else:
        print(f"No results found for query '{args.text}'.")
    print("All operations completed successfully.")


if __name__ == "__main__":
    main()
