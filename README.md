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
Pipeline for linking historical entity mentions to Wikidata, handling NIL (not in Wikidata) entities.

## Pipeline

### 1. Document Preprocessing
```bash
python DocumentPreprocessing/scripts/corpus_prep_paragraphs.py
```
Prepares historical documents for entity extraction.

### 2. Property Extraction
Extract entity properties using either approach:

**RAG (Retrieval-Augmented Generation):**
```bash
python QuestionAnswering/scripts/knowledge\ extraction/rag/RAG_FamilyName.py
```

**Zero-Shot:**
```bash
python QuestionAnswering/scripts/knowledge\ extraction/zero-shot/ZeroShot_FamilyName.py
```

Properties extracted: FamilyName, GivenName, DoB, CoC, Occupation, SexGender

### 3. Entity Linking
Link extracted properties to Wikidata QIDs:
```bash
python ObjectLinking/scripts/linking/entity_linker.py --entity-type family_name --folders path/to/results
```

### 4. Evaluation
Calculate precision, recall, F1:
```bash
python Evaluation/scripts/evaluate_all_properties.py --property FamilyName
```

## Key Features
- **NIL Handling**: Manages entities absent from Wikidata
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
2. Extract properties: `QuestionAnswering/scripts/`
3. Link entities: `ObjectLinking/scripts/linking/`
4. Evaluate: `Evaluation/scripts/`

## Output
- Extracted properties: `QuestionAnswering/data/`
- Linked entities: `ObjectLinking/output/`
- Evaluation reports: `Evaluation/data/`