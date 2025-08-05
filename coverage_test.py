#!/usr/bin/env python3
"""
Coverage Test Script for ShapeNet Vector Database

This script verifies that every embedding file in the shapenet_embedding directory
has a corresponding 3D model in the HuggingFace ShapeNet zip files.

The script checks:
1. For every [synset]-[object]_embedding.npy file in shapenet/shapenet_embedding/
2. That the corresponding [synset].zip exists in hf_shapenet_zips/
3. That the [object] directory exists inside the zip file

File naming pattern:
- Embedding: "02691156-1a04e3eab45ca15dd86060f189eb133_embedding.npy"
- Synset ID: "02691156" (first part before -)
- Object ID: "1a04e3eab45ca15dd86060f189eb133" (second part after - and before _embedding)
- Zip file: "02691156.zip"
- Inside zip: "02691156/1a04e3eab45ca15dd86060f189eb133/" directory
"""

import os
import zipfile
from collections import defaultdict
from pathlib import Path
import time

class ShapeNetCoverageTest:
    def __init__(self, shapenet_dir="/home/shuyuan/ShapenetDB/shapenet", 
                 hf_zips_dir="/home/shuyuan/ShapenetDB/hf_shapenet_zips"):
        self.shapenet_dir = Path(shapenet_dir)
        self.hf_zips_dir = Path(hf_zips_dir)
        self.embedding_dir = self.shapenet_dir / "shapenet_embedding"
        
        # Statistics
        self.stats = {
            'total_embeddings': 0,
            'missing_zip_files': 0,
            'missing_objects_in_zip': 0,
            'valid_pairs': 0,
            'synset_coverage': defaultdict(lambda: {'total': 0, 'missing_zip': 0, 'missing_obj': 0})
        }
        
        # Error tracking
        self.missing_zips = set()
        self.missing_objects = []  # [(synset, object_id, reason)]
        
    def parse_embedding_filename(self, filename):
        """Parse embedding filename to extract synset and object IDs."""
        if not filename.endswith('_embedding.npy'):
            return None, None
            
        # Remove _embedding.npy suffix
        base = filename[:-len('_embedding.npy')]
        
        # Split on first dash to separate synset and object
        if '-' not in base:
            return None, None
            
        synset_id, object_id = base.split('-', 1)
        return synset_id, object_id
    
    def get_zip_contents(self, zip_path):
        """Get the list of directories in a zip file."""
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                # Get all directory names (those ending with /)
                dirs = set()
                for name in zf.namelist():
                    # Extract directory path
                    parts = name.split('/')
                    if len(parts) >= 2:  # synset/object/ structure
                        synset_dir = parts[0]
                        if len(parts) >= 2 and parts[1]:  # has object directory
                            object_dir = parts[1]
                            dirs.add(f"{synset_dir}/{object_dir}")
                return dirs
        except Exception as e:
            print(f"Error reading zip {zip_path}: {e}")
            return set()
    
    def run_coverage_test(self, sample_size=None, verbose=True):
        """
        Run the coverage test.
        
        Args:
            sample_size: If specified, only test this many embedding files (for quick testing)
            verbose: Whether to print progress information
        """
        print("=" * 60)
        print("ShapeNet Vector Database Coverage Test")
        print("=" * 60)
        
        # Get all embedding files
        if verbose:
            print(f"Scanning embedding directory: {self.embedding_dir}")
        
        embedding_files = list(self.embedding_dir.glob("*_embedding.npy"))
        total_files = len(embedding_files)
        
        if sample_size:
            embedding_files = embedding_files[:sample_size]
            print(f"Testing sample of {len(embedding_files)} out of {total_files} embedding files")
        else:
            print(f"Testing all {total_files} embedding files")
        
        # Cache zip contents to avoid repeated reading
        zip_contents_cache = {}
        
        start_time = time.time()
        
        for i, embedding_file in enumerate(embedding_files):
            if verbose and i % 1000 == 0:
                print(f"Progress: {i}/{len(embedding_files)} ({i/len(embedding_files)*100:.1f}%)")
            
            filename = embedding_file.name
            synset_id, object_id = self.parse_embedding_filename(filename)
            
            if not synset_id or not object_id:
                print(f"Warning: Could not parse filename: {filename}")
                continue
                
            self.stats['total_embeddings'] += 1
            self.stats['synset_coverage'][synset_id]['total'] += 1
            
            # Check if zip file exists
            zip_path = self.hf_zips_dir / f"{synset_id}.zip"
            if not zip_path.exists():
                self.stats['missing_zip_files'] += 1
                self.stats['synset_coverage'][synset_id]['missing_zip'] += 1
                self.missing_zips.add(synset_id)
                self.missing_objects.append((synset_id, object_id, "zip_missing"))
                continue
            
            # Get zip contents (use cache)
            if str(zip_path) not in zip_contents_cache:
                zip_contents_cache[str(zip_path)] = self.get_zip_contents(zip_path)
            
            zip_dirs = zip_contents_cache[str(zip_path)]
            expected_path = f"{synset_id}/{object_id}"
            
            if expected_path not in zip_dirs:
                self.stats['missing_objects_in_zip'] += 1
                self.stats['synset_coverage'][synset_id]['missing_obj'] += 1
                self.missing_objects.append((synset_id, object_id, "object_missing"))
            else:
                self.stats['valid_pairs'] += 1
        
        elapsed_time = time.time() - start_time
        self.print_results(elapsed_time, verbose)
        
        return self.stats
    
    def print_results(self, elapsed_time, verbose=True):
        """Print the test results."""
        print("\n" + "=" * 60)
        print("COVERAGE TEST RESULTS")
        print("=" * 60)
        
        total = self.stats['total_embeddings']
        valid = self.stats['valid_pairs']
        missing_zip = self.stats['missing_zip_files']
        missing_obj = self.stats['missing_objects_in_zip']
        
        print(f"Time elapsed: {elapsed_time:.2f} seconds")
        print(f"Total embeddings tested: {total:,}")
        print(f"Valid pairs (embedding + 3D model): {valid:,} ({valid/total*100:.2f}%)")
        print(f"Missing zip files: {missing_zip:,} ({missing_zip/total*100:.2f}%)")
        print(f"Missing objects in zip: {missing_obj:,} ({missing_obj/total*100:.2f}%)")
        
        print(f"\nOverall coverage: {valid/total*100:.2f}%")
        
        if verbose:
            # Print synset-level statistics
            print(f"\nSynset-level Coverage:")
            print("-" * 40)
            synsets = sorted(self.stats['synset_coverage'].keys())
            for synset in synsets:
                data = self.stats['synset_coverage'][synset]
                total_syn = data['total']
                missing_zip_syn = data['missing_zip']
                missing_obj_syn = data['missing_obj']
                valid_syn = total_syn - missing_zip_syn - missing_obj_syn
                coverage = valid_syn / total_syn * 100 if total_syn > 0 else 0
                
                print(f"{synset}: {valid_syn:,}/{total_syn:,} ({coverage:.1f}%) "
                      f"[zip_missing: {missing_zip_syn}, obj_missing: {missing_obj_syn}]")
        
        # Show problematic cases
        if self.missing_zips:
            print(f"\nMissing ZIP files ({len(self.missing_zips)} synsets):")
            for synset in sorted(self.missing_zips):
                print(f"  - {synset}.zip")
        
        if missing_obj > 0:
            print(f"\nSample missing objects (showing first 10):")
            obj_missing = [item for item in self.missing_objects if item[2] == "object_missing"]
            for synset, obj_id, reason in obj_missing[:10]:
                print(f"  - {synset}/{obj_id}")
            if len(obj_missing) > 10:
                print(f"  ... and {len(obj_missing) - 10} more")
    
    def save_detailed_report(self, output_file="coverage_report.txt"):
        """Save a detailed report to file."""
        output_path = Path(output_file)
        
        with open(output_path, 'w') as f:
            f.write("ShapeNet Vector Database Coverage Report\n")
            f.write("=" * 50 + "\n\n")
            
            # Summary
            total = self.stats['total_embeddings']
            valid = self.stats['valid_pairs']
            f.write(f"Summary:\n")
            f.write(f"  Total embeddings: {total:,}\n")
            f.write(f"  Valid pairs: {valid:,} ({valid/total*100:.2f}%)\n")
            f.write(f"  Missing zip files: {self.stats['missing_zip_files']:,}\n")
            f.write(f"  Missing objects: {self.stats['missing_objects_in_zip']:,}\n\n")
            
            # Missing zip files
            if self.missing_zips:
                f.write("Missing ZIP files:\n")
                for synset in sorted(self.missing_zips):
                    f.write(f"  {synset}.zip\n")
                f.write("\n")
            
            # Missing objects
            if self.missing_objects:
                f.write("Missing objects:\n")
                obj_missing = [item for item in self.missing_objects if item[2] == "object_missing"]
                for synset, obj_id, reason in obj_missing:
                    f.write(f"  {synset}/{obj_id}\n")
        
        print(f"Detailed report saved to: {output_path}")


def main():
    """Main function to run the coverage test."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test coverage between ShapeNet embeddings and 3D models")
    parser.add_argument("--sample", type=int, help="Test only a sample of N embedding files")
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")
    parser.add_argument("--report", type=str, help="Save detailed report to file")
    
    args = parser.parse_args()
    
    # Initialize and run test
    tester = ShapeNetCoverageTest()
    
    try:
        results = tester.run_coverage_test(
            sample_size=args.sample, 
            verbose=not args.quiet
        )
        
        if args.report:
            tester.save_detailed_report(args.report)
        
        # Return appropriate exit code
        total = results['total_embeddings']
        valid = results['valid_pairs']
        coverage_percent = valid / total * 100 if total > 0 else 0
        
        if coverage_percent == 100.0:
            print("\n✅ Perfect coverage! All embeddings have corresponding 3D models.")
            return 0
        elif coverage_percent >= 95.0:
            print(f"\n✅ Excellent coverage ({coverage_percent:.2f}%)")
            return 0
        elif coverage_percent >= 90.0:
            print(f"\n⚠️  Good coverage ({coverage_percent:.2f}%) but some models missing")
            return 1
        else:
            print(f"\n❌ Poor coverage ({coverage_percent:.2f}%) - significant missing models")
            return 2
            
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
        return 130
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
