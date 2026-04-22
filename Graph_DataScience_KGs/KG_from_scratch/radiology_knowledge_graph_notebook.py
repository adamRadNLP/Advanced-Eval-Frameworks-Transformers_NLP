# %% [markdown]
# # Radiology Report Knowledge Graph Builder
# **Simplified notebook by:** Adam Lang (adapted for local Jupyter use)
# **For:** Nivi
# **Date:** April 2026
#
# ## What This Notebook Does
# Extracts a **knowledge graph** from radiology report text using:
# 1. **GliNER-BioMed** -- extracts clinical entities (anatomy, observations, diseases)
# 2. **GliREL** -- extracts typed relationships between entities (located_at, suggestive_of, modify)
# 3. **NetworkX** -- builds a directed graph from the entities and relationships
# 4. **pyvis** -- creates an interactive HTML visualization you can explore in your browser
#
# ## Pipeline
# ```
# Report Text --> GliNER-BioMed (entities) --> spaCy (tokenization) --> GliREL (relationships)
#            --> NetworkX (graph) --> pyvis (interactive visualization)
# ```
#
# ## RadGraph Schema (default labels)
# **Entities:**
# - Anatomy definitely present (ANAT-DP)
# - Observation definitely present (OBS-DP)
# - Observation definitely absent (OBS-DA) -- negation!
# - Observation uncertain (OBS-U)
#
# **Relationships:**
# - `suggestive_of`: OBS --> OBS (diagnostic inference)
# - `located_at`: OBS --> ANAT (anatomical localization)
# - `modify`: Any --> Any (descriptor modification)
#
# ## Important Notes
# - spaCy tokenization is **required** for GliREL. GliREL operates at the token level,
#   so spaCy converts GliNER's character-based entity spans to token indices.
# - GliREL's `ner=` parameter (not `entities=`) is the correct API.
# - GliREL returns token LISTS for head/tail text, not strings. Must join them.
# - Threshold of 0.05 works well for radiology. Default 0.3 is too strict.

# %% [markdown]
# ## Step 1: Install Dependencies
# Run this cell once. Restart kernel after.

# %%
# Install all dependencies
# !pip install gliner spacy networkx pyvis pandas torch --quiet
# !pip install git+https://github.com/jackboyla/GLiREL.git --quiet
# !python -m spacy download en_core_web_sm --quiet
# print("All dependencies installed. Restart kernel now.")

# %% [markdown]
# ## Step 2: Imports and Setup

# %%
import os
import json
import time
from typing import Dict, List, Tuple, Any

import pandas as pd
import networkx as nx
import spacy
import torch

from gliner import GLiNER
from glirel import GLiREL
from pyvis.network import Network

print("=" * 60)
print("RADIOLOGY KNOWLEDGE GRAPH BUILDER")
print("=" * 60)
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Load spaCy -- REQUIRED for GliREL
nlp = spacy.load("en_core_web_sm")
print("spaCy loaded")
print("=" * 60)

# %% [markdown]
# ## Step 3: Configuration
# Edit these settings to customize the pipeline.

# %%
# ============================================================================
# CONFIGURATION -- edit these to customize
# ============================================================================

# --- Entity labels (RadGraph schema) ---
# These are the default RadGraph labels. You can add more here.
ENTITY_LABELS = [
    "Anatomy definitely present",
    "Observation definitely present",
    "Observation definitely absent",
    "Observation uncertain",
]

# --- Optional: RadGraph2 extended labels ---
# Uncomment these to add RadGraph2 change entities:
# ENTITY_LABELS += [
#     "Disease definitely present",
#     "Disease definitely absent",
#     "Disease uncertain",
#     "Procedure definitely present",
#     "Procedure definitely absent",
#     "Procedure uncertain",
#     "Change no change",               # CHAN-NC
#     "Change confirmed appearing",     # CHAN-CON-AP
#     "Change confirmed worsening",     # CHAN-CON-WOR
#     "Change confirmed improving",     # CHAN-CON-IMP
#     "Change confirmed resolved",      # CHAN-CON-RES
#     "Change device appearing",        # CHAN-DEV-AP
#     "Change device placement",        # CHAN-DEV-PLACE
#     "Change device disappearing",     # CHAN-DEV-DISA
# ]

# --- Relationship schema (RadGraph) ---
RELATION_LABELS = ["suggestive_of", "located_at", "modify"]

# --- Thresholds ---
GLINER_THRESHOLD = 0.3   # entity confidence (0.3 = good default)
GLIREL_THRESHOLD = 0.05  # relationship confidence (0.05 = tuned for radiology, 0.3 is too strict!)

# --- Models ---
GLINER_MODEL = "Ihor/gliner-biomed-large-v1.0"
GLIREL_MODEL = "jackboyla/glirel-large-v0"

print(f"Entity labels: {len(ENTITY_LABELS)}")
for label in ENTITY_LABELS:
    print(f"  - {label}")
print(f"Relation labels: {RELATION_LABELS}")
print(f"Thresholds: entity={GLINER_THRESHOLD}, relation={GLIREL_THRESHOLD}")

# %% [markdown]
# ## Step 4: Load Models
# This loads both GliNER-BioMed and GliREL onto GPU.
# First run will download models (~1-2 GB total).

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading models on: {device}")

# Load GliNER-BioMed (entity extraction)
print(f"Loading GliNER-BioMed: {GLINER_MODEL}...")
gliner_model = GLiNER.from_pretrained(GLINER_MODEL)
gliner_model = gliner_model.to(device)
print("GliNER loaded")

# Load GliREL (relationship extraction)
print(f"Loading GliREL: {GLIREL_MODEL}...")
glirel_model = GLiREL.from_pretrained(GLIREL_MODEL)
glirel_model = glirel_model.to(device)
print("GliREL loaded")

print(f"\nBoth models ready on {device}")

# %% [markdown]
# ## Step 5: Load Your Data
# Load radiology reports from CSV. Expects a column with report text.

# %%
# ============================================================================
# LOAD DATA -- edit the path and column name to match your data
# ============================================================================

DATA_PATH = "your_reports.csv"  # <-- change this
TEXT_COLUMN = "findings"        # <-- change this to your text column name

# Load CSV
df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} reports from {DATA_PATH}")
print(f"Columns: {list(df.columns)}")
print(f"\nFirst report preview:")
print(df[TEXT_COLUMN].iloc[0][:500])

# %% [markdown]
# ## Step 6: Core Extraction Function
# This is the key function that chains GliNER --> spaCy --> GliREL.
#
# **Critical implementation notes (from debugging):**
# - GliREL needs spaCy tokens, not raw text
# - GliREL's parameter is `ner=`, not `entities=`
# - GliREL returns token LISTS for head_text/tail_text, must join them
# - GliREL's relation key is `label`, not `relation`
# - Threshold 0.05 works for radiology (default 0.3 is too strict)

# %%
def extract_entities_and_relations(
    text: str,
    gliner_model,
    glirel_model,
    entity_labels: List[str] = ENTITY_LABELS,
    relation_labels: List[str] = RELATION_LABELS,
    entity_threshold: float = GLINER_THRESHOLD,
    relation_threshold: float = GLIREL_THRESHOLD,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Extract entities and relationships from radiology report text.

    Pipeline: GliNER (entities) -> spaCy (tokenization) -> GliREL (relationships)

    Args:
        text: radiology report text (findings, impression, etc.)
        gliner_model: loaded GliNER model
        glirel_model: loaded GliREL model
        entity_labels: list of entity type labels for GliNER
        relation_labels: list of relationship type labels for GliREL
        entity_threshold: minimum confidence for entities (default 0.3)
        relation_threshold: minimum confidence for relations (default 0.05)

    Returns:
        (entities, relationships) where each is a list of dicts
    """

    # ------------------------------------------------------------------
    # STEP 1: Extract entities with GliNER
    # ------------------------------------------------------------------
    try:
        raw_entities = gliner_model.predict_entities(
            text, entity_labels, threshold=entity_threshold
        )
        entities = [
            {
                "text": ent["text"],
                "label": ent["label"],
                "start": ent["start"],
                "end": ent["end"],
                "score": ent.get("score", 1.0),
            }
            for ent in raw_entities
        ]
    except Exception as e:
        print(f"  Entity extraction failed: {e}")
        return [], []

    # ------------------------------------------------------------------
    # STEP 2: Extract relationships with GliREL
    # ------------------------------------------------------------------
    relationships = []

    # Need at least 2 entities for relationships
    if len(entities) < 2:
        return entities, relationships

    try:
        # Tokenize with spaCy -- REQUIRED for GliREL
        doc = nlp(text)
        tokens = [token.text for token in doc]

        # Convert GliNER char-span entities to GliREL token-span format
        # GliREL format: [start_token_idx, end_token_idx, LABEL, text]
        ner_for_glirel = []
        for entity in entities:
            start_char = entity["start"]
            end_char = entity["end"]

            # Map character positions to token positions
            start_token = None
            end_token = None
            for i, token in enumerate(doc):
                if token.idx <= start_char < token.idx + len(token.text):
                    start_token = i
                if token.idx < end_char <= token.idx + len(token.text):
                    end_token = i
                    break

            if start_token is not None and end_token is not None:
                ner_for_glirel.append([
                    start_token,
                    end_token,
                    entity["label"].upper(),  # GliREL expects uppercase
                    entity["text"],
                ])

        # Run GliREL if we have enough mapped entities
        if len(ner_for_glirel) >= 2:
            raw_relations = glirel_model.predict_relations(
                tokens,              # token list, NOT raw text
                relation_labels,     # relationship type labels
                threshold=relation_threshold,
                ner=ner_for_glirel,  # parameter is 'ner', NOT 'entities'
                top_k=10,
            )

            if raw_relations:
                for rel in raw_relations:
                    # GliREL returns token LISTS, not strings
                    # e.g. head_text: ['cardiac', 'silhouette'] -> 'cardiac silhouette'
                    head_tokens = rel.get("head_text", [])
                    tail_tokens = rel.get("tail_text", [])

                    if not head_tokens or not tail_tokens:
                        continue

                    head_text = (
                        " ".join(head_tokens) if isinstance(head_tokens, list)
                        else str(head_tokens)
                    )
                    tail_text = (
                        " ".join(tail_tokens) if isinstance(tail_tokens, list)
                        else str(tail_tokens)
                    )

                    relationships.append({
                        "head": head_text,
                        "tail": tail_text,
                        "type": rel.get("label", "unknown"),  # key is 'label', not 'relation'
                        "score": rel.get("score", 0.0),
                    })

    except Exception as e:
        print(f"  Relationship extraction failed: {e}")

    return entities, relationships

# %% [markdown]
# ## Step 7: Test on a Single Report
# Run extraction on one report to verify everything works before batch processing.

# %%
# Pick the first report to test
test_text = df[TEXT_COLUMN].iloc[0]
print(f"Test report ({len(test_text)} chars):")
print(test_text[:300])
print("...")

# Run extraction
start = time.time()
entities, relations = extract_entities_and_relations(test_text, gliner_model, glirel_model)
elapsed = time.time() - start

print(f"\nExtraction complete in {elapsed:.2f}s")
print(f"Entities: {len(entities)}")
print(f"Relationships: {len(relations)}")

# Show entities
print("\n--- ENTITIES ---")
for ent in entities:
    print(f"  [{ent['label']}] {ent['text']} (score: {ent['score']:.3f})")

# Show relationships
print("\n--- RELATIONSHIPS ---")
if relations:
    for rel in relations:
        print(f"  {rel['head']} --[{rel['type']}]--> {rel['tail']} (score: {rel['score']:.3f})")
else:
    print("  No relationships found.")
    print("  Tip: try lowering GLIREL_THRESHOLD (currently {GLIREL_THRESHOLD})")
    print("  Tip: ensure report text has enough clinical detail")

# %% [markdown]
# ## Step 8: Batch Extraction (All Reports)
# Extract entities and relationships from every report in the dataset.
#
# **Optional: Sentence-aware chunking (Step 8b)**
# If your reports are long (>500 tokens), GliREL has a 512-token limit.
# Enable chunking below to split long reports at sentence boundaries
# with 10-20% overlap, then merge results with deduplication.

# %%
# ============================================================================
# OPTIONAL: Sentence-Aware Chunking for Long Reports
# Set USE_CHUNKING = True if your reports exceed ~500 tokens
# ============================================================================
USE_CHUNKING = False  # <-- set to True for long reports

def chunk_text_by_sentences(text: str, max_tokens: int = 480, overlap_tokens: int = 50):
    """
    Split text into chunks at sentence boundaries with overlap.

    - Preserves sentence integrity (no mid-sentence splits)
    - 10-20% overlap between chunks prevents entity loss at boundaries
    - Falls back to token-level splitting for very long sentences

    Args:
        text: input text
        max_tokens: max tokens per chunk (480 = safe margin for GliREL's 512 limit)
        overlap_tokens: minimum token overlap between chunks

    Returns:
        list of chunk strings
    """
    doc = nlp(text)
    tokens = [token.text for token in doc]

    # If text fits in one chunk, return as-is
    if len(tokens) <= max_tokens:
        return [text]

    sentences = list(doc.sents)
    chunks = []
    current_sents = []
    current_token_count = 0

    for sent in sentences:
        sent_token_count = len([t for t in sent])

        if current_token_count + sent_token_count > max_tokens and current_sents:
            # Save current chunk
            chunk_text = " ".join([s.text for s in current_sents])
            chunks.append(chunk_text)

            # Start new chunk with overlap (keep last few sentences)
            overlap_count = 0
            overlap_sents = []
            for s in reversed(current_sents):
                s_tokens = len([t for t in s])
                if overlap_count + s_tokens <= overlap_tokens:
                    overlap_sents.insert(0, s)
                    overlap_count += s_tokens
                else:
                    break

            current_sents = overlap_sents
            current_token_count = overlap_count

        current_sents.append(sent)
        current_token_count += sent_token_count

    # Last chunk
    if current_sents:
        chunk_text = " ".join([s.text for s in current_sents])
        chunks.append(chunk_text)

    return chunks


def extract_with_chunking(text, gliner_model, glirel_model):
    """Extract from long text using sentence-aware chunking + merge."""
    chunks = chunk_text_by_sentences(text)

    all_entities = []
    all_relations = []
    seen_entities = {}  # canonical text -> entity dict (dedup)

    for chunk in chunks:
        ents, rels = extract_entities_and_relations(chunk, gliner_model, glirel_model)

        # Deduplicate entities across chunks (keep highest score)
        for ent in ents:
            key = ent["text"].lower().strip()
            if key not in seen_entities or ent["score"] > seen_entities[key]["score"]:
                seen_entities[key] = ent

        all_relations.extend(rels)

    # Deduplicate relationships
    seen_rels = set()
    unique_rels = []
    for rel in all_relations:
        key = (rel["head"].lower(), rel["tail"].lower(), rel["type"])
        if key not in seen_rels:
            seen_rels.add(key)
            unique_rels.append(rel)

    return list(seen_entities.values()), unique_rels

# %%
all_results = {}
total_entities = 0
total_relations = 0
start_time = time.time()

for idx, row in df.iterrows():
    text = row[TEXT_COLUMN]

    # Skip empty text
    if not isinstance(text, str) or len(text.strip()) < 10:
        continue

    if USE_CHUNKING:
        entities, relations = extract_with_chunking(text, gliner_model, glirel_model)
    else:
        entities, relations = extract_entities_and_relations(
            text, gliner_model, glirel_model
        )

    all_results[idx] = {
        "text": text,
        "entities": entities,
        "relationships": relations,
    }

    total_entities += len(entities)
    total_relations += len(relations)

    if (idx + 1) % 10 == 0:
        print(f"Processed {idx + 1}/{len(df)} reports "
              f"({total_entities} entities, {total_relations} relations)")

elapsed = (time.time() - start_time) / 60
print(f"\nBatch extraction complete: {len(all_results)} reports in {elapsed:.1f} min")
print(f"Total entities: {total_entities}")
print(f"Total relationships: {total_relations}")
print(f"Chunking: {'ON' if USE_CHUNKING else 'OFF'}")

# %% [markdown]
# ## Step 8c: Entity Resolution (SERF Methodology) -- OPTIONAL
#
# **What this does:** Different reports (or even different sentences in the same report)
# may refer to the same entity with different text:
# - "cardiac silhouette" vs "heart shadow" vs "the cardiac silhouette"
# - "right lung" vs "right pulmonary" vs "rt lung"
#
# Entity resolution collapses these into a single canonical node in the graph.
# Without it, the graph has duplicate nodes for the same real-world entity.
#
# **Method (SERF -- Semantic Entity Resolution Framework):**
# 1. Embed all entities with a radiology-specific model (RadEval ModernBERT)
# 2. Find similar entity pairs using kNN
# 3. Use an LLM (via DSPy) to confirm whether similar pairs are truly the same entity
# 4. Merge confirmed matches using connected components
#
# **Toggle:** Set `USE_ENTITY_RESOLUTION = True` to enable.
# Compare your graph with and without resolution to see the difference.
#
# **Requirements:** `pip install sentence-transformers scikit-learn dspy-ai`
# Plus API access to an LLM for the DSPy confirmation step.

# %%
USE_ENTITY_RESOLUTION = False  # <-- set to True to enable

# ============================================================================
# Skip this cell if USE_ENTITY_RESOLUTION is False
# ============================================================================
if USE_ENTITY_RESOLUTION:
    from transformers import AutoTokenizer, AutoModel
    from sklearn.neighbors import NearestNeighbors
    import numpy as np
    import dspy

    # ------------------------------------------------------------------
    # STEP 1: Collect all unique entity texts across all reports
    # ------------------------------------------------------------------
    all_entity_texts = set()
    for idx, result in all_results.items():
        for ent in result["entities"]:
            all_entity_texts.add(ent["text"].lower().strip())

    entity_list = sorted(all_entity_texts)
    print(f"Unique entities to resolve: {len(entity_list)}")

    # ------------------------------------------------------------------
    # STEP 2: Embed with RadEval ModernBERT (radiology-specific)
    # ------------------------------------------------------------------
    print("Loading RadEval ModernBERT for entity embeddings...")
    radeval_tokenizer = AutoTokenizer.from_pretrained("IAMJB/RadEvalModernBERT")
    radeval_model = AutoModel.from_pretrained("IAMJB/RadEvalModernBERT")
    radeval_model.to(device)
    radeval_model.eval()

    def get_embeddings(texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Get CLS token embeddings from RadEval ModernBERT."""
        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = radeval_tokenizer(
                batch, return_tensors="pt", padding=True,
                truncation=True, max_length=128
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = radeval_model(**inputs, output_hidden_states=True)
                cls_embs = outputs.hidden_states[-1][:, 0, :]

            all_embs.append(cls_embs.cpu().numpy())

            if (i // batch_size + 1) % 10 == 0:
                print(f"  Embedded {min(i + batch_size, len(texts))}/{len(texts)}")

        return np.vstack(all_embs)

    embeddings = get_embeddings(entity_list)
    # Normalize for cosine similarity
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    print(f"Embeddings shape: {embeddings.shape}")

    # ------------------------------------------------------------------
    # STEP 3: kNN blocking -- find candidate pairs
    # ------------------------------------------------------------------
    K_NEIGHBORS = 6        # self + 5 neighbors
    MAX_DISTANCE = 0.30    # cosine distance threshold (similarity > 0.70)

    nbrs = NearestNeighbors(
        n_neighbors=min(K_NEIGHBORS, len(entity_list)),
        metric="cosine", algorithm="auto", n_jobs=-1
    ).fit(embeddings)

    distances, indices = nbrs.kneighbors(embeddings)

    # Build candidate pairs (above similarity threshold)
    candidate_pairs = []
    for i in range(len(entity_list)):
        for j_idx in range(1, len(indices[i])):  # skip self (index 0)
            j = indices[i][j_idx]
            dist = distances[i][j_idx]
            if dist < MAX_DISTANCE and i < j:  # avoid duplicates
                candidate_pairs.append((entity_list[i], entity_list[j], 1.0 - dist))

    print(f"Candidate pairs to verify: {len(candidate_pairs)}")

    # ------------------------------------------------------------------
    # STEP 4: LLM verification with DSPy (confirm matches)
    # ------------------------------------------------------------------
    # Configure DSPy with your LLM
    # Uncomment ONE of these:
    # lm = dspy.LM("openai/gpt-4o-mini", api_key="your-key")
    # lm = dspy.LM("ollama_chat/mistral", api_base="http://localhost:11434")
    # dspy.configure(lm=lm)

    class EntityMatcher(dspy.Signature):
        """Determine if two radiology entities refer to the same real-world concept."""
        entity1: str = dspy.InputField(desc="First entity text")
        entity2: str = dspy.InputField(desc="Second entity text")
        are_same: str = dspy.OutputField(desc="yes or no")

    matcher = dspy.Predict(EntityMatcher)

    # Verify each candidate pair
    confirmed_pairs = []
    for ent1, ent2, similarity in candidate_pairs:
        try:
            result = matcher(entity1=ent1, entity2=ent2)
            if result.are_same.strip().lower() == "yes":
                confirmed_pairs.append((ent1, ent2))
        except Exception as e:
            print(f"  LLM verification failed for ({ent1}, {ent2}): {e}")

    print(f"Confirmed matches: {len(confirmed_pairs)}")

    # ------------------------------------------------------------------
    # STEP 5: Build canonical mapping using connected components
    # ------------------------------------------------------------------
    resolution_graph = nx.Graph()
    resolution_graph.add_nodes_from(entity_list)
    resolution_graph.add_edges_from(confirmed_pairs)

    canonical_mapping = {}
    for component in nx.connected_components(resolution_graph):
        # Pick the shortest entity text as canonical name
        canonical = min(component, key=len)
        for entity_text in component:
            canonical_mapping[entity_text] = canonical

    # Count how many entities were merged
    n_merged = len(entity_list) - len(set(canonical_mapping.values()))
    print(f"Entity resolution complete:")
    print(f"  Before: {len(entity_list)} unique entities")
    print(f"  After:  {len(set(canonical_mapping.values()))} canonical entities")
    print(f"  Merged: {n_merged} duplicates resolved")

    # Show some example merges
    merge_groups = {}
    for ent, canonical in canonical_mapping.items():
        if ent != canonical:
            merge_groups.setdefault(canonical, []).append(ent)

    if merge_groups:
        print(f"\nExample merges (showing up to 10):")
        for canonical, variants in list(merge_groups.items())[:10]:
            print(f"  '{canonical}' <- {variants}")

else:
    canonical_mapping = None
    print("Entity resolution: OFF")
    print("Set USE_ENTITY_RESOLUTION = True to enable (compare graph with/without)")

# %% [markdown]
# ## Step 9: Build NetworkX Graph
# Create a directed graph from the extracted entities and relationships.
# If entity resolution is enabled, duplicate entities are collapsed.

# %%
def build_graph(
    entities: List[Dict],
    relationships: List[Dict],
    canonical_mapping: Dict = None,
) -> nx.DiGraph:
    """Build a NetworkX directed graph from entities and relationships.

    If canonical_mapping is provided (from entity resolution),
    duplicate entities are collapsed into single nodes.
    """
    G = nx.DiGraph()

    # Color map for entity types
    color_map = {
        "definitely present": "#4CAF50",  # green
        "definitely absent":  "#F44336",  # red
        "uncertain":          "#FFC107",  # yellow/amber
    }

    # Add nodes
    for i, ent in enumerate(entities):
        # Determine certainty from label
        label_lower = ent["label"].lower()
        if "definitely present" in label_lower:
            certainty = "definitely present"
        elif "definitely absent" in label_lower:
            certainty = "definitely absent"
        elif "uncertain" in label_lower:
            certainty = "uncertain"
        else:
            certainty = "unknown"

        entity_text = ent["text"].lower().strip()

        # Apply entity resolution if available
        display_text = entity_text
        if canonical_mapping:
            display_text = canonical_mapping.get(entity_text, entity_text)

        node_id = f"ent_{display_text}"  # use canonical text as ID to merge duplicates

        # Add node (or update if already exists with higher score)
        if node_id not in G.nodes:
            G.add_node(
                node_id,
                label=display_text,
                entity_type=ent["label"],
                certainty=certainty,
                score=ent["score"],
                color=color_map.get(certainty, "#9E9E9E"),
            )
        elif ent["score"] > G.nodes[node_id].get("score", 0):
            G.nodes[node_id]["score"] = ent["score"]

    # Build text-to-node-id lookup
    text_to_id = {}
    for node_id, data in G.nodes(data=True):
        text_to_id[data["label"]] = node_id

    # Add edges
    for rel in relationships:
        head_text = rel["head"].lower().strip()
        tail_text = rel["tail"].lower().strip()

        # Apply entity resolution to relationship endpoints
        if canonical_mapping:
            head_text = canonical_mapping.get(head_text, head_text)
            tail_text = canonical_mapping.get(tail_text, tail_text)

        head_id = text_to_id.get(head_text)
        tail_id = text_to_id.get(tail_text)

        if head_id and tail_id and head_id != tail_id:
            G.add_edge(
                head_id,
                tail_id,
                relation=rel["type"],
                score=rel["score"],
            )

    return G

# Build graph from first report as a test
test_entities = all_results[0]["entities"]
test_relations = all_results[0]["relationships"]
G_test = build_graph(test_entities, test_relations, canonical_mapping)
print(f"Test graph: {G_test.number_of_nodes()} nodes, {G_test.number_of_edges()} edges")
if canonical_mapping:
    print(f"  (entity resolution applied -- duplicates collapsed)")

# %% [markdown]
# ## Step 10: Interactive Visualization with pyvis
# Creates an HTML file you can open in your browser to explore the graph.

# %%
def visualize_graph_interactive(
    G: nx.DiGraph,
    title: str = "Radiology Knowledge Graph",
    output_file: str = "graph.html",
    height: str = "800px",
    width: str = "100%",
):
    """Create an interactive pyvis visualization from a NetworkX graph."""
    net = Network(
        height=height,
        width=width,
        directed=True,
        notebook=True,
        cdn_resources="remote",
    )

    # Add nodes with colors and labels
    for node_id, data in G.nodes(data=True):
        net.add_node(
            node_id,
            label=data.get("label", node_id),
            color=data.get("color", "#9E9E9E"),
            title=(
                f"Entity: {data.get('label', '')}\n"
                f"Type: {data.get('entity_type', '')}\n"
                f"Certainty: {data.get('certainty', '')}\n"
                f"Score: {data.get('score', 0):.3f}"
            ),
            size=25,
        )

    # Edge color map
    edge_colors = {
        "located_at": "#2196F3",     # blue
        "suggestive_of": "#FF9800",  # orange
        "modify": "#9C27B0",         # purple
    }

    # Add edges with labels and colors
    for source, target, data in G.edges(data=True):
        rel_type = data.get("relation", "unknown")
        net.add_edge(
            source,
            target,
            label=rel_type,
            color=edge_colors.get(rel_type, "#666666"),
            title=f"{rel_type} (score: {data.get('score', 0):.3f})",
            arrows="to",
        )

    # Physics settings for better layout
    net.set_options("""
    {
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -50,
          "centralGravity": 0.01,
          "springLength": 200,
          "springConstant": 0.08
        },
        "solver": "forceAtlas2Based",
        "stabilization": {"iterations": 150}
      },
      "edges": {
        "font": {"size": 10, "align": "middle"},
        "smooth": {"type": "curvedCW", "roundness": 0.2}
      },
      "nodes": {
        "font": {"size": 14}
      }
    }
    """)

    net.save_graph(output_file)
    print(f"Interactive graph saved to: {output_file}")
    print(f"Open in browser to explore. Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    print(f"  Green = present, Red = absent, Yellow = uncertain")
    print(f"  Blue edges = located_at, Orange = suggestive_of, Purple = modify")

    return net


# Visualize the test graph
net = visualize_graph_interactive(G_test, title="Report 1 Knowledge Graph", output_file="report_1_graph.html")

# %% [markdown]
# ## Step 11: Build and Visualize All Reports
# Build a combined "mega-graph" from all reports, or individual graphs per report.

# %%
# Build individual graphs for each report
graphs = {}
for idx, result in all_results.items():
    G = build_graph(result["entities"], result["relationships"], canonical_mapping)
    graphs[idx] = G

# Print summary
print(f"Built {len(graphs)} graphs")
for idx, G in graphs.items():
    print(f"  Report {idx}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# %%
# Build a combined mega-graph (all reports merged)
G_mega = nx.DiGraph()
for idx, G in graphs.items():
    G_mega = nx.compose(G_mega, G)

print(f"\nMega-graph: {G_mega.number_of_nodes()} nodes, {G_mega.number_of_edges()} edges")

# Visualize
visualize_graph_interactive(G_mega, title="All Reports - Mega Graph", output_file="mega_graph.html")

# %% [markdown]
# ## Step 12: Graph Algorithms (Exploratory)
# Run basic graph algorithms to understand the structure.

# %%
def analyze_graph(G: nx.DiGraph, name: str = "Graph"):
    """Run basic graph algorithms and print summary."""
    print(f"\n{'=' * 60}")
    print(f"GRAPH ANALYSIS: {name}")
    print(f"{'=' * 60}")
    print(f"Nodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")

    if G.number_of_nodes() == 0:
        print("Empty graph -- nothing to analyze.")
        return {}

    results = {}

    # PageRank (importance)
    pr = nx.pagerank(G)
    top_pr = sorted(pr.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"\nTop 10 by PageRank (most important entities):")
    for node_id, score in top_pr:
        label = G.nodes[node_id].get("label", node_id)
        print(f"  {label}: {score:.4f}")
    results["pagerank"] = pr

    # Betweenness centrality (bridge entities)
    bc = nx.betweenness_centrality(G)
    top_bc = sorted(bc.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"\nTop 5 by Betweenness Centrality (bridge entities):")
    for node_id, score in top_bc:
        label = G.nodes[node_id].get("label", node_id)
        print(f"  {label}: {score:.4f}")
    results["betweenness"] = bc

    # Connected components
    n_components = nx.number_weakly_connected_components(G)
    print(f"\nWeakly connected components: {n_components}")
    results["n_components"] = n_components

    # Relation type distribution
    rel_counts = {}
    for _, _, data in G.edges(data=True):
        rel_type = data.get("relation", "unknown")
        rel_counts[rel_type] = rel_counts.get(rel_type, 0) + 1
    print(f"\nRelationship distribution:")
    for rel_type, count in sorted(rel_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {rel_type}: {count}")
    results["relation_distribution"] = rel_counts

    # Entity type distribution
    type_counts = {}
    for _, data in G.nodes(data=True):
        etype = data.get("entity_type", "unknown")
        type_counts[etype] = type_counts.get(etype, 0) + 1
    print(f"\nEntity type distribution:")
    for etype, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {etype}: {count}")
    results["entity_distribution"] = type_counts

    return results


# Analyze the mega-graph
results = analyze_graph(G_mega, "Mega-Graph (All Reports)")

# %% [markdown]
# ## Step 13: Export Graph Data
# Save entities, relationships, and graph structure for downstream use (e.g., GraphRAG).

# %%
# Export all entities to CSV
all_entities_flat = []
for idx, result in all_results.items():
    for ent in result["entities"]:
        all_entities_flat.append({
            "report_idx": idx,
            "text": ent["text"],
            "label": ent["label"],
            "score": ent["score"],
        })

entities_df = pd.DataFrame(all_entities_flat)
entities_df.to_csv("extracted_entities.csv", index=False)
print(f"Saved {len(entities_df)} entities to extracted_entities.csv")

# Export all relationships to CSV
all_rels_flat = []
for idx, result in all_results.items():
    for rel in result["relationships"]:
        all_rels_flat.append({
            "report_idx": idx,
            "head": rel["head"],
            "tail": rel["tail"],
            "type": rel["type"],
            "score": rel["score"],
        })

rels_df = pd.DataFrame(all_rels_flat)
rels_df.to_csv("extracted_relationships.csv", index=False)
print(f"Saved {len(rels_df)} relationships to extracted_relationships.csv")

# Export graph as GraphML (can be loaded into Gephi, Neo4j, etc.)
nx.write_graphml(G_mega, "mega_graph.graphml")
print("Saved mega_graph.graphml")

print(f"\nExported files:")
print(f"  extracted_entities.csv -- all entities with types and scores")
print(f"  extracted_relationships.csv -- all relationships with types and scores")
print(f"  mega_graph.graphml -- full graph structure (import into Gephi/Neo4j)")
print(f"  report_1_graph.html -- interactive visualization of first report")
print(f"  mega_graph.html -- interactive visualization of all reports")

# %% [markdown]
# ---
# ## Optional: DSPy Zero-Shot Extraction (LLM-based)
# Use an LLM to extract entities and relationships as a comparison to GliNER/GliREL.
# Requires API access to an LLM (e.g., OpenAI, Anthropic, or a local model).
#
# Uncomment and configure the cells below to try it.

# %%
# # --- DSPy Zero-Shot Entity-Relationship Extraction ---
# # Uncomment this section to use an LLM for extraction
#
# import dspy
# from pydantic import BaseModel, Field
#
# # Configure DSPy with your LLM
# # Option 1: OpenAI
# # lm = dspy.LM("openai/gpt-4o-mini", api_key="your-key-here")
# # Option 2: Local model via vLLM/Ollama
# # lm = dspy.LM("ollama_chat/mistral", api_base="http://localhost:11434")
# # dspy.configure(lm=lm)
#
#
# class Entity(BaseModel):
#     """A clinical entity extracted from a radiology report."""
#     name: str = Field(description="The entity text as it appears in the report")
#     type: str = Field(description="Entity type: Anatomy, Observation, Disease, or Procedure")
#     certainty: str = Field(description="Certainty: definitely_present, definitely_absent, or uncertain")
#
#
# class Relationship(BaseModel):
#     """A relationship between two clinical entities."""
#     source: str = Field(description="The source entity name")
#     target: str = Field(description="The target entity name")
#     relation: str = Field(description="Relationship type: located_at, suggestive_of, or modify")
#     evidence: str = Field(description="Short quote from the text supporting this relationship")
#
#
# class RadGraphExtraction(dspy.Signature):
#     """Extract clinical entities and their relationships from a radiology report.
#     Use the RadGraph schema:
#     - Entity types: Anatomy, Observation, Disease, Procedure
#     - Certainty: definitely_present, definitely_absent, uncertain
#     - Relationships: located_at (observation->anatomy), suggestive_of (observation->observation), modify (any->any)
#     """
#     report_text: str = dspy.InputField(desc="The radiology report text")
#     entities: list[Entity] = dspy.OutputField(desc="List of extracted clinical entities")
#     relationships: list[Relationship] = dspy.OutputField(desc="List of relationships between entities")
#
#
# extractor = dspy.Predict(RadGraphExtraction)
#
# # Test on one report
# test_text = df[TEXT_COLUMN].iloc[0]
# result = extractor(report_text=test_text)
#
# print("DSPy Entities:")
# for ent in result.entities:
#     print(f"  [{ent.type} / {ent.certainty}] {ent.name}")
#
# print("\nDSPy Relationships:")
# for rel in result.relationships:
#     print(f"  {rel.source} --[{rel.relation}]--> {rel.target}")
#     print(f"    Evidence: {rel.evidence}")
