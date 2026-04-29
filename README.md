# CitationHub

Explore influential papers, citation networks, citation contexts, and knowledge graphs across multidisciplinary scientific domains.

[![Live Demo](https://img.shields.io/badge/Live-Demo-blue)](https://citationdatabase.streamlit.app)
[![Dataset](https://img.shields.io/badge/Dataset-Zenodo-orange)](https://zenodo.org/records/18536895)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Space-yellow)](https://huggingface.co/spaces/Daniel0315/cithub_website)

---

##  Public Access

### Demo Website

https://citationdatabase.streamlit.app

### Hugging Face Space

https://huggingface.co/spaces/Daniel0315/cithub_website

### Hugging Face Dataset 

**CitationHub: A Large-Scale Multi-disciplinary Citation Context Database**
https://zenodo.org/records/18536895

---

##  Overview

CitationHub is a large-scale citation context database and interactive exploration platform designed to support:

* Citation Intent Classification
* Citation Recommendation
* Scholarly Retrieval
* Knowledge Graph Construction
* Contextual Citation Evaluation
* Research Trend Analysis
* Scientific Discovery Support

Unlike traditional citation databases that treat citations as simple links between papers, CitationHub preserves:

* citation context (actual citing sentence)
* citation intent labels (why the citation was made)
* multidisciplinary field information
* co-citation relationships
* citation event structures
* knowledge graph representation

This enables more fine-grained and explainable scholarly analysis.

---

##  System Interface

![CitationHub Demo System](./assets/citationhub_demo.png)

The platform provides several interactive modules for citation exploration and knowledge graph analysis.

---

## Main Features

### 1. Dashboard

Users can explore:

* highly cited seed papers
* citation event statistics
* citation intent distributions
* field distributions
* related citing papers
* co-cited seed papers

Supported filters include:

* Title or DOI
* Field
* Country
* Journal
* Citation Year

This enables efficient exploration of citation behaviors across scientific disciplines.

---

### 2. Citation Network

Visual exploration of relationships among:

* seed papers
* citing papers
* co-cited papers

This helps identify citation flow and scholarly influence propagation.

---

### 3. Knowledge Graph

CitationHub transforms citation events into structured scholarly knowledge graphs for:

* semantic querying
* graph-based reasoning
* citation event understanding
* explainable scholarly discovery

---

### 4. Geographic Map

Global visualization of citation distributions by:

* country
* institution
* affiliation

This supports international collaboration analysis.

---

### 5. Analytics

Advanced analytics for:

* citation intent comparison
* field-level analysis
* journal-level patterns
* temporal citation trends

---

##  Key Statistics

CitationHub currently contains:

| Category          |  Count |
| ----------------- | -----: |
| Seed Papers       | 23,479 |
| Citation Events   | 1.83M+ |
| Citing Papers     | 1.44M+ |
| Authors           | 16,839 |
| Countries         |    108 |
| Scientific Fields |     21 |

This makes CitationHub one of the largest multidisciplinary citation-context-aware resources.

---

##  Citation Intent Categories

CitationHub supports 7 major citation intent categories:

* Background
* Uses
* Similarities
* Motivation
* Differences
* Future Work
* Extends

These intent labels provide controllable and interpretable signals for:

* intent-aware citation retrieval
* reranking systems
* citation recommendation
* scholarly evaluation

---

##  Research Applications

### Intent-aware Citation Retrieval

Intent-conditioned citation recommendation and selective reranking.

### Scholarly Knowledge Graph Construction

Citation events → RDF triples → semantic scholarly graphs.

### Contextual Citation Evaluation

Moving beyond simple citation counts toward semantic impact measurement.

### Research Trend Discovery

Field-specific citation behaviors and knowledge evolution analysis.

---



##  Repository Structure

```text
CitationHub/
├── app.py
├── requirements.txt
└── README.md
```

---

##  Citation

If you use CitationHub in your research, please cite:

```bibtex
@dataset{citationhub2026,
  title={CitationHub: A Large-Scale Multi-disciplinary Citation Context Database},
  author={Nam, Seohyun and others},
  year={2026},
  publisher={Zenodo},
  doi={10.5281/zenodo.18536895}
}
```


---

##  Future Work

We are expanding CitationHub toward:

* full citation event ontology
* LLM-based citation reasoning
* agentic scholarly discovery systems
* explainable citation recommendation
* benchmark datasets for top-tier citation retrieval research

---
