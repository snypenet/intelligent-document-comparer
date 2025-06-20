#!/usr/bin/env python3
"""
Export ChromaDB data for a specific index type and file ID to a file.
"""

import argparse
import json
import os
from pathlib import Path
import chromadb

def export_chromadb_data(file_id: str, index_type: str, output_file: str = None):
    """
    Export ChromaDB data for a specific file_id and index_type.
    
    Args:
        file_id: The file ID to export data for
        index_type: The index type (document_a or document_b)
        output_file: Output file path (optional, will auto-generate if not provided)
    """
    if not output_file:
        output_file = f"chromadb_export_{file_id}_{index_type}.json"
    
    print(f"Exporting ChromaDB data for file_id: {file_id}, index_type: {index_type}")
    print(f"Output file: {output_file}")
    
    try:
        # Connect to ChromaDB directly
        client = chromadb.PersistentClient(path="./chroma_db")
        collection = client.get_or_create_collection("document_embeddings")
        
        # Get all data from the collection
        all_data = collection.get()
        
        if not all_data["documents"]:
            print(f"No data found in ChromaDB collection")
            return
        
        # Filter by file_id and index_type
        filtered_docs = []
        filtered_metas = []
        filtered_ids = []
        
        for i, (doc_id, doc_text, meta) in enumerate(zip(all_data["ids"], all_data["documents"], all_data["metadatas"])):
            if meta.get("file_id") == file_id and meta.get("index_type") == index_type:
                filtered_docs.append(doc_text)
                filtered_metas.append(meta)
                filtered_ids.append(doc_id)
        
        if not filtered_docs:
            print(f"No data found for file_id: {file_id}, index_type: {index_type}")
            return
        
        print(f"Found {len(filtered_docs)} chunks")
        
        # Prepare data for export
        export_data = {
            "file_id": file_id,
            "index_type": index_type,
            "total_chunks": len(filtered_docs),
            "chunks": []
        }
        
        for i, (doc_id, doc_text, meta) in enumerate(zip(filtered_ids, filtered_docs, filtered_metas)):
            chunk_data = {
                "id": doc_id,
                "text": doc_text,
                "metadata": meta,
                "chunk_number": i + 1
            }
            export_data["chunks"].append(chunk_data)
        
        # Write to file
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"Successfully exported {len(filtered_docs)} chunks to {output_file}")
        
        # Print summary
        print("\nSummary:")
        print(f"  Total chunks: {len(filtered_docs)}")
        
        # Count chunks with assigned topics
        chunks_with_topics = sum(1 for meta in filtered_metas if meta.get("assigned_topics"))
        print(f"  Chunks with assigned topics: {chunks_with_topics}")
        
        # Show unique assigned topics
        all_assigned_topics = set()
        for meta in filtered_metas:
            assigned_topics = meta.get("assigned_topics", "")
            if assigned_topics:
                topics = assigned_topics.split("||||")
                all_assigned_topics.update(topics)
        
        print(f"  Unique assigned topics: {len(all_assigned_topics)}")
        if all_assigned_topics:
            print("  Topics found:")
            for topic in sorted(all_assigned_topics):
                print(f"    - {topic}")
        
        # Show page ranges
        page_ranges = set()
        for meta in filtered_metas:
            pages = meta.get("pages", "")
            if pages:
                page_ranges.add(pages)
        
        print(f"  Page ranges: {len(page_ranges)}")
        if page_ranges:
            print("  Pages:")
            for pages in sorted(page_ranges, key=lambda x: [int(p) for p in x.split(",") if p.isdigit()]):
                print(f"    - {pages}")
        
    except Exception as e:
        print(f"Error exporting data: {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description="Export ChromaDB data for a specific file_id and index_type")
    parser.add_argument("--file-id", required=True, help="File ID to export data for")
    parser.add_argument("--index-type", required=True, choices=["document_a", "document_b"], 
                       help="Index type (document_a or document_b)")
    parser.add_argument("--output", help="Output file path (optional)")
    
    args = parser.parse_args()
    
    export_chromadb_data(args.file_id, args.index_type, args.output)

if __name__ == "__main__":
    main() 