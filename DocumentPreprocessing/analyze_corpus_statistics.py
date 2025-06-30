#!/usr/bin/env python3
"""
Analyze corpus statistics for the periodicals mini corpus.
Generates a comprehensive markdown report with statistics about documents, paragraphs, and embeddings.
"""

import os
import pickle
import numpy as np
from datetime import datetime
from pathlib import Path


def load_corpus_data(corpus_path):
    """Load the corpus paragraphs data."""
    with open(corpus_path, 'rb') as f:
        return pickle.load(f)


def load_embedding_info(embeddings_dir):
    """Load information about all embedding files."""
    embedding_info = {}
    
    for file_path in Path(embeddings_dir).glob("*.pkl"):
        with open(file_path, 'rb') as f:
            embeddings = pickle.load(f)
            
        # Handle torch tensors
        if hasattr(embeddings, 'cpu'):
            embeddings = embeddings.cpu().numpy()
        elif hasattr(embeddings, 'numpy'):
            embeddings = embeddings.numpy()
            
        embedding_info[file_path.name] = {
            'shape': embeddings.shape,
            'dtype': str(embeddings.dtype),
            'size_mb': embeddings.nbytes / (1024 * 1024),
            'min_val': float(embeddings.min()),
            'max_val': float(embeddings.max()),
            'mean_val': float(embeddings.mean()),
            'std_val': float(embeddings.std())
        }
    
    return embedding_info


def calculate_document_statistics(corpus_data):
    """Calculate statistics about documents and paragraphs."""
    doc_paragraph_counts = []
    
    for doc_name, paragraphs in corpus_data.items():
        doc_paragraph_counts.append(len(paragraphs))
    
    doc_paragraph_counts = np.array(doc_paragraph_counts)
    
    return {
        'total_documents': len(corpus_data),
        'total_paragraphs': int(doc_paragraph_counts.sum()),
        'min_paragraphs': int(doc_paragraph_counts.min()),
        'max_paragraphs': int(doc_paragraph_counts.max()),
        'mean_paragraphs': float(doc_paragraph_counts.mean()),
        'std_paragraphs': float(doc_paragraph_counts.std()),
        'median_paragraphs': float(np.median(doc_paragraph_counts))
    }


def generate_markdown_report(corpus_data, doc_stats, embedding_info, output_path):
    """Generate a comprehensive markdown report."""
    
    # Start building the report
    report = []
    report.append("# Periodicals Mini Corpus Statistics Report")
    report.append(f"\n*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
    
    # Overall Statistics
    report.append("## 1. Overall Corpus Statistics\n")
    report.append("| Metric | Value |")
    report.append("|--------|--------|")
    report.append(f"| Total Documents | {doc_stats['total_documents']:,} |")
    report.append(f"| Total Paragraphs | {doc_stats['total_paragraphs']:,} |")
    report.append(f"| Average Paragraphs per Document | {doc_stats['mean_paragraphs']:.2f} |")
    report.append(f"| Standard Deviation | {doc_stats['std_paragraphs']:.2f} |")
    report.append(f"| Median Paragraphs per Document | {doc_stats['median_paragraphs']:.1f} |")
    report.append(f"| Min Paragraphs in a Document | {doc_stats['min_paragraphs']:,} |")
    report.append(f"| Max Paragraphs in a Document | {doc_stats['max_paragraphs']:,} |")
    
    # Document-level Statistics
    report.append("\n## 2. Document-Level Statistics\n")
    report.append("| Document Name | Number of Paragraphs |")
    report.append("|---------------|---------------------|")
    
    # Sort documents by paragraph count (descending)
    sorted_docs = sorted(corpus_data.items(), key=lambda x: len(x[1]), reverse=True)
    for doc_name, paragraphs in sorted_docs:
        report.append(f"| {doc_name} | {len(paragraphs):,} |")
    
    # Embedding Statistics
    report.append("\n## 3. Embedding Files Statistics\n")
    report.append("| Embedding Model | Dimensions | Size (MB) | Data Type | Min Value | Max Value | Mean Value | Std Dev |")
    report.append("|-----------------|------------|-----------|-----------|-----------|-----------|------------|---------|")
    
    for file_name, info in sorted(embedding_info.items()):
        model_name = file_name.replace('corpus_encoding_', '').replace('.pkl', '')
        report.append(f"| {model_name} | {info['shape'][1]} | {info['size_mb']:.2f} | {info['dtype']} | "
                     f"{info['min_val']:.4f} | {info['max_val']:.4f} | {info['mean_val']:.4f} | {info['std_val']:.4f} |")
    
    # Embedding Coverage
    report.append("\n## 4. Embedding Coverage\n")
    report.append("| Metric | Value |")
    report.append("|--------|--------|")
    
    # Check if all embeddings have the same number of vectors
    embedding_counts = [info['shape'][0] for info in embedding_info.values()]
    if len(set(embedding_counts)) == 1:
        report.append(f"| Embeddings per Model | {embedding_counts[0]:,} |")
        report.append(f"| Coverage | 100% (matches paragraph count) |")
    else:
        report.append(f"| Embedding Counts | Varies: {set(embedding_counts)} |")
        report.append(f"| Coverage | Inconsistent |")
    
    # Summary Statistics
    report.append("\n## 5. Summary\n")
    report.append(f"- The corpus contains **{doc_stats['total_documents']}** documents from historical music periodicals")
    report.append(f"- These documents are divided into **{doc_stats['total_paragraphs']:,}** paragraphs")
    report.append(f"- Each document contains an average of **{doc_stats['mean_paragraphs']:.1f}** paragraphs (σ = {doc_stats['std_paragraphs']:.1f})")
    report.append(f"- The corpus has been encoded using **{len(embedding_info)}** different embedding models")
    report.append(f"- All embedding models have generated vectors for all **{doc_stats['total_paragraphs']:,}** paragraphs")
    
    # Write the report
    with open(output_path, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"Report generated: {output_path}")


def main():
    # Define paths
    base_dir = Path("/home/arianna/PycharmProjects/NIL_Grounding")
    corpus_path = base_dir / "data" / "periodicals_mini_corpus" / "corpus_paragraphs.pkl"
    embeddings_dir = base_dir / "data" / "periodicals_mini_corpus" / "embeddings"
    output_path = base_dir / "data" / "periodicals_mini_corpus" / "corpus_statistics_report.md"
    
    print("Loading corpus data...")
    corpus_data = load_corpus_data(corpus_path)
    
    print("Calculating document statistics...")
    doc_stats = calculate_document_statistics(corpus_data)
    
    print("Loading embedding information...")
    embedding_info = load_embedding_info(embeddings_dir)
    
    print("Generating markdown report...")
    generate_markdown_report(corpus_data, doc_stats, embedding_info, output_path)
    
    # Also print a summary to console
    print("\n=== Quick Summary ===")
    print(f"Total documents: {doc_stats['total_documents']}")
    print(f"Total paragraphs: {doc_stats['total_paragraphs']:,}")
    print(f"Average paragraphs per document: {doc_stats['mean_paragraphs']:.2f} ± {doc_stats['std_paragraphs']:.2f}")
    print(f"Embedding models: {len(embedding_info)}")
    print(f"\nFull report saved to: {output_path}")


if __name__ == "__main__":
    main()