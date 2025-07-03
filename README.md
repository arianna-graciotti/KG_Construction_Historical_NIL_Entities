# KG Construction for Historical NIL Entities

## Directory Structure
```
KG_Construction_Historical_NIL_Entities/
├── Datasets/
│   └── NIL-KG.csv                    # Historical entity dataset
├── DocumentPreprocessing/
│   └── scripts/                      # Corpus preparation tools
├── QuestionAnswering/
│   └── scripts/
│       └── knowledge extraction/
│           ├── rag/                  # RAG-based property extraction
│           └── zero-shot/            # Zero-shot property extraction
├── ObjectLinking/
│   ├── lookup_tables/                # Wikidata entity mappings
│   └── scripts/
│       ├── linking/                  # Entity linking pipeline
│       └── wikidata_properties/      # Wikidata extraction tools
├── PropertyClustering/
│   └── scripts/                      # Property clustering analysis
└── Evaluation/
    ├── data/                         # Evaluation results by property
    └── scripts/                      # Evaluation metrics calculation
```

## Overview
Pipeline that uses retrieval-augmented generative AI to extract structured knowledge about NIL entities from historical documents and creates Wikidata-compliant KGs.

## Pipeline

### 1. Document Preprocessing

Prepares historical documents for entity extraction.

### 2. Question Answering
Casts property's extraction as a Question Answering task in two settings:

- **RAG (Retrieval-Augmented Generation):**
- **Zero-Shot:**


Properties extracted (selected through **Propert Clustering**): FamilyName, GivenName, DoB, CoC, Occupation, SexGender

### 3. Object Linking
Link extracted properties to Wikidata QIDs.

### 4. Evaluation
Calculate precision, recall, F1 of the properties extracted against the gold standard KGs (in **Datasets**)

## Key Features
- **NIL entities KG construction**: Builds KGs for entities absent from Wikidata (NIL) leveraging information extracted from specialised corpora
- **Multi-Model**: Tests 6 LLMs (GPT-4o, LLAMA, Gemma, Phi-3, Mixtral, Qwen)
- **Multi-Retriever**: BM25, Contriever, GTR, BGE, Instructor
- **Robust Evaluation**: Separate NIL/QID evaluation, year-based DoB matching, multi-class occupation evaluation

## Requirements
- Python 3.8+
- PyTorch
- Transformers
- Sentence-Transformers
- pandas, numpy
- litellm (for LLM API access)

## Quick Start
1. Prepare corpus: `DocumentPreprocessing/scripts/`
2. Select properties: `DocumentPreprocessing/scripts/`
3. Extract properties: `QuestionAnswering/scripts/`
4. Link entities: `ObjectLinking/scripts/linking/`
5. Evaluate: `Evaluation/scripts/`

## Output
- Extracted properties: `QuestionAnswering/data/`
- Linked entities: `ObjectLinking/output/`
- Evaluation reports: `Evaluation/data/`
