# Databricks notebook source
# MAGIC %md
# MAGIC # Polars with GraphFrames on Patient Records
# MAGIC * Notebook by Adam Lang
# MAGIC * Date: 8/1/2025
# MAGIC
# MAGIC # Overview
# MAGIC In this notebook we will do the following:
# MAGIC
# MAGIC 1. Ingest patient record from `json_sopr`
# MAGIC
# MAGIC 2. Flatten JSON
# MAGIC
# MAGIC 3. Use DSPy for zero-shot entity-relationship extraction and resolution.
# MAGIC   * LLM used: `databricks/databricks-claude-4-sonnet` (databricks hosted model)
# MAGIC
# MAGIC 4. Move extracted entities and relationships into a Polars dataframe.
# MAGIC
# MAGIC 5. Insert data into GraphFrames.
# MAGIC   * Future goal is to try and use an open-source Graph Database (e.g. kuzu is defunct now would use ladybug instead or Neptune but you don't need a graph database).
# MAGIC
# MAGIC 6. Visualization using Graphistry.
# MAGIC   * Graphistry was chosen as it excels at visualizing and working with LARGE graphs on spark. 
# MAGIC
# MAGIC
# MAGIC ## Important Notes
# MAGIC * UDF dataframe was tried in another notebook but known issues with LLM on worker vs. driver nodes was blocker.
# MAGIC * Polars is a high-performance, open-source DataFrame library that provides fast and efficient data processing capabilities, especially valuable when dealing with large datasets. 
# MAGIC   * It was built with Rust, a language known for its speed and safety, and offers Python, R, and NodeJS wrappers. 
# MAGIC   * For more about Polars see documention. 

# COMMAND ----------

# MAGIC %md
# MAGIC # PyGraphistry
# MAGIC * This is an API for visualizing large graphs in databricks and spark. 
# MAGIC * You need an API key to make it work. You can sign up here: https://www.graphistry.com/

# COMMAND ----------

# Uncomment and run first time or#  have databricks admin install graphistry python library:#  https://docs.databricks.com/en/libraries/package-repositories.html#pypi-package#
%pip install graphistry

## restart python kernel
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC Import and test if Graphistry is working

# COMMAND ----------

import graphistry  # if not yet available, install pygraphistry and/or restart Python kernel using the cells above
graphistry.__version__

# COMMAND ----------

# MAGIC %md
# MAGIC ## Authentication for Graphistry
# MAGIC * This is a critical step for pygraphistry as it's a hosted visualization platform.
# MAGIC * You need to register with your Graphistry account credentials, ideally fetched from Databricks Secrets for security

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create Secret Scope

# COMMAND ----------

## Step 1 - Create Scope
from databricks.sdk import WorkspaceClient
w = WorkspaceClient()
scope_name = "adam_secret_scope"
# List all secret scopes
existing_scopes = w.secrets.list_scopes()
# Check if the scope exists in the list
if scope_name in [scope.name for scope in existing_scopes]:
    print(f"Secret scope '{scope_name}' already exists.")
else:
    try:
        w.secrets.create_scope(scope=scope_name)
        print(f"Secret scope '{scope_name}' created successfully.")
    except Exception as e:
        print(f"An error occurred while creating the scope: {e}")

# COMMAND ----------

## Step 2: Store the API Key in the Secret Scope
# Store the first key
w.secrets.put_secret(
    scope="adam_secret_scope",
    key="personal_key_id",
    string_value="<key goes here>" ## you need a personal key id and secret from graphistry to fill in here
)

# Store the second key
w.secrets.put_secret(
    scope="adam_secret_scope",
    key="personal_key_secret",
    string_value="<key goes here>" ## you need a personal key id and secret from graphistry to fill in here
)

# COMMAND ----------

## register keys
graphistry.register(api=3,
                    personal_key_id=dbutils.secrets.get(scope="adam_secret_scope", key="personal_key_id"),
                    personal_key_secret=dbutils.secrets.get(scope="adam_secret_scope", key="personal_key_secret"),
                    protocol='https',
                    server='hub.graphistry.com')

# COMMAND ----------

# MAGIC %md
# MAGIC # Install Dependencies

# COMMAND ----------

# Install required packages
%pip install dspy-ai pydantic networkx matplotlib mlflow polars
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC # 0. Imports

# COMMAND ----------

# --- Cluster Library Installation Check ---
# IMPORTANT: Ensure GraphFrames is installed on your Databricks cluster.
# Go to your cluster config -> Libraries -> Install New -> Maven
# Coordinates: graphframes:graphframes:0.8.3-spark3.5-s_2.12
# (Verify your exact Spark and Scala version and update the GraphFrames version if needed)
import os
import json
import time
from typing import Dict, List, Optional, Set, Union
from datetime import date, datetime
from pathlib import Path

import dspy
from pydantic import BaseModel, Field, ValidationError

import networkx as nx
import matplotlib.pyplot as plt
import polars as pl

# Spark imports (needed for final GraphFrame conversion and reading Delta tables)
from pyspark.sql.functions import col, from_json, collect_list, concat_ws, array_join, explode
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, MapType, LongType, IntegerType, DateType

from graphframes import GraphFrame
import graphistry

# COMMAND ----------

# MAGIC %md
# MAGIC # 1. Setup Databricks LLM model

# COMMAND ----------

# --- Databricks-specific setup for DSPy with hosted model ---
# This only needs to be set for the driver node now
DATABRICKS_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
DATABRICKS_HOST = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
DATABRICKS_MODEL_NAME = "databricks/databricks-claude-sonnet-4"  # Or your specific endpoint name

# Set environment variables for the driver only (no UDF workers needed with Polars)
os.environ['DATABRICKS_TOKEN'] = DATABRICKS_TOKEN
os.environ['DATABRICKS_HOST'] = DATABRICKS_HOST
os.environ['DATABRICKS_MODEL_NAME'] = DATABRICKS_MODEL_NAME

print(f"Databricks Model Serving configuration captured (Model: {DATABRICKS_MODEL_NAME})")

# COMMAND ----------

# MAGIC %md
# MAGIC # 2. Pydantic Models Setup

# COMMAND ----------

# --- Pydantic Models for Knowledge Graph Structure ---

RelationshipType = str
EntityType = str

class ClinicalEntity(BaseModel):
    """Represents a unique clinical entity with its canonical name and type."""
    type: EntityType = Field(..., description="The type of the clinical entity (e.g., 'Condition', 'Medication').")
    name: str = Field(..., description="The canonical name of the clinical entity.")

class ResolvedEntities(BaseModel):
    """A collection of all identified unique clinical entities."""
    entity_map: Dict[str, ClinicalEntity] = Field(
        ..., description="A mapping of all identified entity variations (keys) to their resolved, canonical name and type (values)."
    )

class ClinicalRelationship(BaseModel):
    """A single relationship between two clinical entities with supporting evidence."""
    source_entity_name: str = Field(..., description="The canonical name of the source clinical entity.")
    relationship_type: RelationshipType = Field(..., description="The type of relationship.")
    target_entity_name: str = Field(..., description="The canonical name of the target clinical entity.")
    evidence: str = Field(..., description="A direct quote or short summary from the provided text that explicitly supports this relationship.")

class ExtractedAndResolvedClinicalRelationships(BaseModel):
    """The full structured output containing resolved entities and their relationships."""
    resolved_entities: ResolvedEntities = Field(
        ..., description="All unique clinical entities identified, their canonical forms, and types."
    )
    relationships: List[ClinicalRelationship] = Field(
        ..., description="A list of clinical entities and their relationships, using resolved entity names."
    )


# COMMAND ----------

# MAGIC %md
# MAGIC # 3. DSPy Setup
# MAGIC * Below we will setup DSPy signatures. 
# MAGIC * A quick note about DSPy: for now we are not going to use DSPy optimizers as we don't have "golden" data to create a test set, but will add in future. 
# MAGIC
# MAGIC ## 3a. DSPy Signatures Setup

# COMMAND ----------

# --- DSPy Signatures ---

class ClinicalInformationExtraction(dspy.Signature):
    """
    Extracts and resolves clinical entities and their relationships from patient records.
    The output must strictly follow the specified JSON schema for 'resolved_entities' and 'relationships'.
    """
    patient_record_text: str = dspy.InputField(
        description="The full patient record text to be processed."
    )
    json_output: str = dspy.OutputField(
        description=(
            "A JSON object with two top-level keys: 'resolved_entities' and 'relationships'.\n"
            "\n"
            "1. **'resolved_entities'**: A dictionary with a single key 'entity_map'.\n"
            "   - **'entity_map'**: A dictionary where keys are *all mentions/variations* of an entity found in the text, and values are objects with 'type' (e.g., 'Condition', 'Medication', 'Symptom', 'Procedure', 'Demographic', 'Body Part', 'Vaccine') and 'name' (the single, consistent, canonical name for that entity).\n"
            "   Example for 'entity_map' value:\n"
            "   \"Lyme disease\": {\"type\": \"Condition\", \"name\": \"Lyme disease\"}\n"
            "   \"vomiting\": {\"type\": \"Symptom\", \"name\": \"acute vomiting\"}\n"
            "   \"Amoxicillin\": {\"type\": \"Medication\", \"name\": \"Amoxicillin\"}\n"
            "   \"right hind leg\": {\"type\": \"Body Part\", \"name\": \"right hind leg\"}\n"
            "\n"
            "2. **'relationships'**: A list of relationship objects. Each object MUST have the following structure:\n"
            "   - **'source_entity_name'**: (string) The *canonical name* of the source entity from 'entity_map'.\n"
            "   - **'relationship_type'**: (string) The type of relationship. Express as a concise verb or verb phrase (e.g., 'diagnosed with', 'treated with', 'causes', 'is a symptom of', 'located in').\n"
            "   - **'target_entity_name'**: (string) The *canonical name* of the target entity from 'entity_map'.\n"
            "   - **'evidence'**: (string) A direct quote or concise summary from the provided text that explicitly supports this relationship.\n"
            "\n"
            "**Crucially, identify ALL relevant entities and ALL their relationships within the provided patient record text.** Ensure all entity names in relationships precisely match a canonical name defined in the 'entity_map' (i.e., they must be values of the 'name' field in 'entity_map').\n"
            "Example for a 'relationships' list item:\n"
            "{\n"
            "  \"source_entity_name\": \"Duke\",\n"
            "  \"relationship_type\": \"diagnosed with\",\n"
            "  \"target_entity_name\": \"Anaplasma\",\n"
            "  \"evidence\": \"The patient has been diagnosed with several chronic conditions, including Anaplasma and Lyme disease.\"\n"
            "}"
        ),
    )

# --- DSPy Signature for Relationship Type Normalization ---
class RelationshipTypeNormalizer(dspy.Signature):
    """Given a raw relationship phrase, provide a single, canonical, and concise term for that relationship."""
    raw_phrase: str = dspy.InputField(desc="A raw relationship phrase extracted from text.")
    canonical_type: str = dspy.OutputField(desc="The single, canonical, and concise term for this relationship.")

# --- DSPy Signature for Entity Type Normalization ---
class EntityTypeNormalizer(dspy.Signature):
    """Given a raw entity type phrase, provide a single, canonical, and concise term for that entity type."""
    raw_type_phrase: str = dspy.InputField(desc="A raw entity type phrase extracted from text.")
    canonical_type: str = dspy.OutputField(desc="The single, canonical, and concise term for this entity type.")

class ClinicalKnowledgeGraphExtractor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.extract_json_string = dspy.ChainOfThought(ClinicalInformationExtraction)
        self.normalize_relationship = dspy.ChainOfThought(RelationshipTypeNormalizer)
        self.normalize_entity_type = dspy.ChainOfThought(EntityTypeNormalizer)

    def forward(self, patient_record_text: str) -> ExtractedAndResolvedClinicalRelationships:
        prediction = self.extract_json_string(patient_record_text=patient_record_text)
        json_string = prediction.json_output

        print("\n--- Raw LLM JSON Output ---")
        print(json_string)
        print("---------------------------\n")

        try:
            # First attempt: Parse the JSON as is
            parsed_data = json.loads(json_string)
            print("Successfully parsed JSON. Attempting Pydantic validation...")

            # Add JSON repair if needed
            if not all(key in parsed_data for key in ["resolved_entities", "relationships"]):
                print("WARNING: Missing required top-level keys. Attempting to fix the structure...")
                # Simple fix attempt
                if "entity_map" in parsed_data and isinstance(parsed_data["entity_map"], dict):
                    parsed_data = {
                        "resolved_entities": {"entity_map": parsed_data["entity_map"]},
                        "relationships": parsed_data.get("relationships", [])
                    }
                    
            extracted_data = ExtractedAndResolvedClinicalRelationships.model_validate(parsed_data)
            print("Pydantic validation successful.")

            # Entity Type Normalization
            unique_raw_entity_types: Set[str] = {
                entity.type for entity_map_val in extracted_data.resolved_entities.entity_map.values()
                for entity in [entity_map_val]
            }
            print(f"\nUnique raw entity types extracted before normalization: {unique_raw_entity_types}")

            canonical_entity_type_map: Dict[str, str] = {}
            for raw_entity_type in unique_raw_entity_types:
                print(f"Normalizing entity type: '{raw_entity_type}'")
                normalization_prediction = self.normalize_entity_type(raw_type_phrase=raw_entity_type)
                canonical_form = normalization_prediction.canonical_type.strip()
                canonical_entity_type_map[raw_entity_type] = canonical_form
            
            print(f"Canonical entity type map created: {canonical_entity_type_map}")

            for mention, entity in extracted_data.resolved_entities.entity_map.items():
                entity.type = canonical_entity_type_map.get(
                    entity.type, entity.type
                )

            # Relationship Type Normalization
            unique_raw_relationships: Set[str] = {rel.relationship_type for rel in extracted_data.relationships}
            print(f"\nUnique raw relationships extracted before normalization: {unique_raw_relationships}")

            canonical_relationship_map: Dict[str, str] = {}
            for raw_rel_type in unique_raw_relationships:
                print(f"Normalizing relationship type: '{raw_rel_type}'")
                normalization_prediction = self.normalize_relationship(raw_phrase=raw_rel_type)
                canonical_form = normalization_prediction.canonical_type.strip()
                canonical_relationship_map[raw_rel_type] = canonical_form
            
            print(f"Canonical relationship map created: {canonical_relationship_map}")

            for rel in extracted_data.relationships:
                rel.relationship_type = canonical_relationship_map.get(
                    rel.relationship_type, rel.relationship_type
                )
            
            return extracted_data
            
        except json.JSONDecodeError as e:
            print(f"\n!!! JSON DECODE ERROR !!!")
            print(f"Problem at: line {e.lineno} column {e.colno} (char {e.pos})")
            print(f"Expected: '{e.msg}'")
            
            # Attempt to repair the JSON
            from json.decoder import JSONDecodeError
            try:
                # Simple repair: replace single quotes with double quotes, fix missing commas
                repaired_json = json_string.replace("'", "\"").replace("},\n  ]", "}\n  ]")
                parsed_data = json.loads(repaired_json)
                print("Successfully repaired and parsed JSON!")
                extracted_data = ExtractedAndResolvedClinicalRelationships.model_validate(parsed_data)
                return extracted_data
            except (JSONDecodeError, ValidationError) as repair_error:
                print(f"Repair attempt failed: {repair_error}")
                raise ValueError(f"LLM did not return valid JSON: {e}\nRaw output: {json_string}")
                
        except ValidationError as e:
            print(f"\n!!! PYDANTIC VALIDATION ERROR !!!")
            print(f"Validation errors: {e.errors()}")
            raise ValueError(f"LLM output JSON does not match Pydantic schema: {e}")
            
        except Exception as e:
            print(f"\n!!! UNEXPECTED ERROR !!!")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {e}")
            raise RuntimeError(f"An unexpected error occurred during JSON parsing or validation: {e}")


# COMMAND ----------

# MAGIC %md
# MAGIC ## 3b. Configure DSPy

# COMMAND ----------

# --- Function to Configure DSPy ---
def configure_dspy():
    """Configures DSPy to use the Databricks-hosted LLM."""
    databricks_token = os.environ.get('DATABRICKS_TOKEN')
    if not databricks_token:
        raise ValueError("DATABRICKS_TOKEN is not set in environment variables.")

    databricks_workspace_url = os.environ.get('DATABRICKS_HOST')
    databricks_endpoint_name = os.environ.get('DATABRICKS_MODEL_NAME')

    api_base_for_litellm = f"{databricks_workspace_url}/serving-endpoints"

    llm = dspy.LM(
        model=databricks_endpoint_name, # Remove the f"openai/" prefix"
        api_key=databricks_token,
        api_base=api_base_for_litellm,
        max_tokens=32000,
        temperature=0.0,
    )
    dspy.settings.configure(lm=llm)
    print(f"DSPy configured to use Databricks LLM: {databricks_workspace_url} with endpoint {databricks_endpoint_name}")
    return ClinicalKnowledgeGraphExtractor()

# COMMAND ----------

# MAGIC %md
# MAGIC # 4. Data Loading and Processing -- using Polars

# COMMAND ----------

# --- Data Loading and Processing using Polars ---

def load_and_process_patient_data(num_records: int = 1):
    """Load and process patient data from Delta table into Polars DataFrame."""
    catalog_name = "llm_sandbox"
    schema_name = "aiphs"
    table_name = "json_sopr"
    full_table_path = f"{catalog_name}.{schema_name}.{table_name}"
    
    print(f"Loading data from Delta table: {full_table_path} (limiting to {num_records} records)")
    
    # JSON schema for parsing (unchanged)
    json_schema_full = StructType([
        StructField("demographics", StructType([
            StructField("animal", StructType([
                StructField("animal_rowkey", StringType(), True),
                StructField("animal_name", StringType(), True),
                StructField("species", StringType(), True),
                StructField("breed", StringType(), True),
                StructField("status", StringType(), True),
                StructField("birth_date", StringType(), True), 
                StructField("gender", StringType(), True),
                StructField("age", IntegerType(), True),
                StructField("vcp_patient_rowkeys", ArrayType(StringType()), True) 
            ]), True),
            StructField("practice", StructType([
                StructField("practice_rowkey", StringType(), True),
                StructField("practice_name", StringType(), True),
                StructField("pims", StringType(), True),
                StructField("state_province", StringType(), True),
                StructField("country", StringType(), True)
            ]), True),
            StructField("owner", StringType(), True) 
        ]), True),
        StructField("patient_summary", StructType([
            StructField("patient_summary_text", StringType(), True),
            StructField("patient_summary_stats", StructType([
                StructField("appointments", IntegerType(), True),
                StructField("conditions", IntegerType(), True),
                StructField("lab_jobs", IntegerType(), True),
                StructField("lab_observations", IntegerType(), True),
                StructField("medical_notes", IntegerType(), True),
                StructField("prescriptions", IntegerType(), True),
                StructField("transactions", IntegerType(), True),
                StructField("observations", IntegerType(), True),
                StructField("vaccinations", IntegerType(), True),
                StructField("pathology_reports", IntegerType(), True),
                StructField("visits", IntegerType(), True)
            ]), True)
        ]), True),
        StructField("visit_summaries", ArrayType(StructType([
            StructField("visit_start_date", StringType(), True),
            StructField("visit_first_day", StringType(), True),
            StructField("visit_last_day", StringType(), True),
            StructField("visit_type", StringType(), True),
            StructField("visit_count_lab_observations", IntegerType(), True),
            StructField("visit_count_observations", IntegerType(), True),
            StructField("visit_count_prescriptions", IntegerType(), True),
            StructField("visit_count_transactions", IntegerType(), True),
            StructField("visit_count_vaccinations", IntegerType(), True),
            StructField("visit_count_medical_notes", IntegerType(), True),
            StructField("visit_count_pathology_reports", IntegerType(), True),
            StructField("visit_summary", StringType(), True)
        ])), True)
    ])

    try:
        # Use Spark to load from Delta and transform data
        raw_df = spark.read.table(full_table_path)
        
        processed_df = raw_df.withColumn("parsed_json", from_json(col("json_sopr"), json_schema_full)) \
                             .select(
                                 col("pims_animal_rowkey").alias("patient_id"), 
                                 col("parsed_json.patient_summary.patient_summary_text").alias("patient_summary_text"),
                                 col("parsed_json.visit_summaries").alias("visit_summaries_array"),
                                 col("parsed_json.demographics").alias("demographics")
                             )
        
        patient_data_df = processed_df.filter(col("visit_summaries_array").isNotNull()) \
                                 .withColumn("visit_summary_text", explode(col("visit_summaries_array").getField("visit_summary"))) \
                                 .groupBy("patient_id", "patient_summary_text", "demographics") \
                                 .agg(collect_list("visit_summary_text").alias("all_visit_summaries_list")) \
                                 .withColumn(
                                     "full_patient_text",
                                     concat_ws("\n", col("patient_summary_text"), array_join(col("all_visit_summaries_list"), "\n"))
                                 ) \
                                 .select("patient_id", "full_patient_text", "demographics") \
                                 .limit(num_records)

        # Convert to Polars
        polars_df = pl.from_pandas(patient_data_df.toPandas())
        
        actual_num_records = len(polars_df)
        print(f"Successfully loaded and processed {actual_num_records} records into Polars DataFrame.")
        
        if actual_num_records == 0:
            print("WARNING: No records loaded. Check table name, column names, and data presence.")
            return None
            
        return polars_df
        
    except Exception as e:
        print(f"ERROR: Could not load or process data from table '{full_table_path}'. Error: {e}")
        raise

# COMMAND ----------

# MAGIC %md
# MAGIC # 5. Knowledge Graph Extraction and Building -- using Polars
# MAGIC
# MAGIC ## 5a. Knowledge Graph Extraction Code
# MAGIC * The code below will extract the graph.

# COMMAND ----------

# --- Knowledge Graph Extraction and Building Function with Polars ---

def extract_kg_for_patient(patient_id: str, patient_text: str, demographics: dict, extractor):
    """Extract knowledge graph data for a single patient."""
    print(f"Processing patient {patient_id}, text length: {len(patient_text)}")
    
    if not patient_text or len(patient_text.strip()) == 0:
        print(f"Empty patient text for patient {patient_id}. Returning empty graph.")
        return {"patient_id": patient_id, "kg_data": {"resolved_entities": {"entity_map": {}}, "relationships": []}}
    
    try:
        # Flatten the patient data into a more coherent text
        # If demographics is available, include it
        if demographics:
            animal_data = demographics.get('animal', {})
            practice_data = demographics.get('practice', {})
            
            demographics_text = f"Patient Demographics:\n" \
                        f"  Name: {animal_data.get('animal_name', 'N/A')}\n" \
                        f"  Species: {animal_data.get('species', 'N/A')}\n" \
                        f"  Breed: {animal_data.get('breed', 'N/A')}\n" \
                        f"  Age: {animal_data.get('age', 'N/A')}\n" \
                        f"  Gender: {animal_data.get('gender', 'N/A')}\n" \
                        f"  Clinic: {practice_data.get('practice_name', 'N/A')}\n" \
                        f"  Location: {practice_data.get('state_province', 'N/A')}, {practice_data.get('country', 'N/A')}\n"
                        
            full_text = f"--- Patient Record ---\n{demographics_text}\n{patient_text}"
        else:
            full_text = patient_text
        
        # Extract KG data using the extractor
        start_time = time.time()
        extracted_data_obj = extractor(patient_record_text=full_text)
        end_time = time.time()
        
        print(f"Extraction for patient {patient_id} completed in {end_time - start_time:.2f} seconds")
        print(f"Extracted {len(extracted_data_obj.resolved_entities.entity_map)} entities and {len(extracted_data_obj.relationships)} relationships")
        
        # Convert to dictionary format
        kg_data = extracted_data_obj.model_dump(mode='json')
        return {"patient_id": patient_id, "kg_data": kg_data}
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"ERROR processing patient {patient_id}: {e}")
        return {"patient_id": patient_id, "kg_data": {"resolved_entities": {"entity_map": {}}, "relationships": []}}

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5b. Building a knowledge graph
# MAGIC * After extraction, the code below builds a graph in GraphFrames.

# COMMAND ----------

# --- Graph Building Function with Polars ---

def build_patient_graph(polars_df, visualize=True, save_to_delta=False):
    """Build GraphFrame knowledge graph from patient data using Polars."""
    
    print("=" * 60)
    print("BUILDING PATIENT KNOWLEDGE GRAPH WITH POLARS")
    print("=" * 60)
    
    start_time = time.time()
    
    # Step 1: Initialize DSPy extractor
    print("Step 1: Initializing DSPy extractor...")
    extractor = configure_dspy()
    
    # Step 2: Process each patient with the extractor
    print("Step 2: Extracting knowledge graph data using DSPy...")
    kg_start_time = time.time()
    
    kg_results = []
    
    for row in polars_df.iter_rows(named=True):
        patient_id = row['patient_id']
        full_patient_text = row['full_patient_text']
        demographics = row.get('demographics', {})
        
        # Extract KG data
        result = extract_kg_for_patient(patient_id, full_patient_text, demographics, extractor)
        kg_results.append(result)
    
    # Convert results to a Polars DataFrame
    kg_result_df = pl.DataFrame(kg_results)
    
    kg_end_time = time.time()
    kg_extraction_time = kg_end_time - kg_start_time
    
    print(f" Knowledge graph extraction completed for {len(kg_result_df)} records")
    print(f"  Time taken: {kg_extraction_time:.2f} seconds ({kg_extraction_time/60:.2f} minutes)")
    
    # Step 3: Extract nodes and edges
    print("\nStep 3: Creating nodes and edges DataFrames...")
    graph_start_time = time.time()
    
    all_nodes = []
    all_edges = []
    
    # Process each row to extract nodes and edges with error handling
    for row in kg_result_df.iter_rows(named=True):
        kg_data = row['kg_data']
        
        # Extract nodes from entity_map with error handling
        if 'resolved_entities' in kg_data and 'entity_map' in kg_data['resolved_entities']:
            entity_map = kg_data['resolved_entities']['entity_map']
            for mention, entity in entity_map.items():
                # Check if entity is not None and has the required keys
                if entity is not None and isinstance(entity, dict) and 'name' in entity and 'type' in entity:
                    node = {
                        'id': entity['name'],  # Use canonical name as ID
                        'node_type': 'Entity',
                        'subtype': entity['type'],
                        'attributes': {'name': entity['name'], 'type': entity['type']}
                    }
                    all_nodes.append(node)
                else:
                    print(f"Warning: Invalid entity structure for mention '{mention}': {entity}")
        
        # Extract edges from relationships with error handling
        if 'relationships' in kg_data:
            relationships = kg_data['relationships']
            for rel in relationships:
                # Check if rel has all required fields
                if (rel is not None and isinstance(rel, dict) and 
                    'source_entity_name' in rel and 
                    'target_entity_name' in rel and 
                    'relationship_type' in rel and 
                    'evidence' in rel):
                    
                    edge = {
                        'src': rel['source_entity_name'],
                        'dst': rel['target_entity_name'],
                        'relationship': rel['relationship_type'],
                        'evidence': rel['evidence']
                    }
                    all_edges.append(edge)
                else:
                    print(f"Warning: Invalid relationship structure: {rel}")
    
    # Check if we have valid data to create DataFrames
    if not all_nodes:
        print("Warning: No valid nodes extracted. Cannot create graph.")
        return None
    
    # Convert to Polars DataFrames with unique nodes
    nodes_df = pl.DataFrame(all_nodes).unique(subset=['id'])
    
    if all_edges:
        edges_df = pl.DataFrame(all_edges)
        # Group edges by src, dst, relationship and concatenate evidence
        edges_df = edges_df.group_by(['src', 'dst', 'relationship']).agg(
            pl.concat_str('evidence', separator='\n\n').alias('evidence')
        )
    else:
        print("Warning: No valid edges extracted. Creating empty edges DataFrame.")
        edges_df = pl.DataFrame(schema={'src': str, 'dst': str, 'relationship': str, 'evidence': str})
    
    # Convert Polars DataFrames to Spark DataFrames for GraphFrame
    spark_nodes_df = spark.createDataFrame(nodes_df.to_pandas())
    spark_edges_df = spark.createDataFrame(edges_df.to_pandas())
    
    # Create GraphFrame
    kg_graph = GraphFrame(spark_nodes_df, spark_edges_df)
    
    # Get counts
    num_vertices = kg_graph.vertices.count()
    num_edges = kg_graph.edges.count()
    
    graph_end_time = time.time()
    graph_creation_time = graph_end_time - graph_start_time
    
    print(f" GraphFrame created successfully")
    print(f"  Number of vertices: {num_vertices}")
    print(f"  Number of edges: {num_edges}")
    print(f"  Time taken: {graph_creation_time:.2f} seconds")
    
    # Total time
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"TOTAL GRAPH BUILDING TIME: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"{'='*60}")
    
    # Breakdown
    print(f"\nTime Breakdown:")
    print(f"  Knowledge Graph Extraction: {kg_extraction_time:.2f}s ({kg_extraction_time/total_time*100:.1f}%)")
    print(f"  Graph Structure Creation:   {graph_creation_time:.2f}s ({graph_creation_time/total_time*100:.1f}%)")
    
    # Display sample data
    print(f"\n--- Sample Nodes ---")
    print(nodes_df.head(5))
    print(f"\n--- Sample Edges ---")
    print(edges_df.head(5))
    
    # Visualization
    if visualize and num_vertices > 0:
        print(f"\n--- Creating Graphistry Visualization ---")
        visualize_graph(spark_nodes_df, spark_edges_df)
    
    # Save to Delta tables
    if save_to_delta:
        print(f"\n--- Saving to Delta Tables ---")
        save_graph_to_delta(spark_nodes_df, spark_edges_df)
    
    return kg_graph, nodes_df, edges_df, {
        'total_time': total_time,
        'kg_extraction_time': kg_extraction_time,
        'graph_creation_time': graph_creation_time,
        'num_vertices': num_vertices,
        'num_edges': num_edges
    }

# COMMAND ----------

# MAGIC %md
# MAGIC # 6. Visualization Function Setup -- using Graphistry

# COMMAND ----------

# --- Visualization Function ---
def visualize_graph(nodes_df, edges_df):
    """Create Graphistry visualization."""
    try:
        # Setup Graphistry
        graphistry.register(
            api=3,
            personal_key_id=dbutils.secrets.get(scope="adam_secret_scope", key="personal_key_id"),
            personal_key_secret=dbutils.secrets.get(scope="adam_secret_scope", key="personal_key_secret"),
            protocol='https',
            server='hub.graphistry.com'
        )
        
        # Convert to Pandas if they're Spark DataFrames
        if hasattr(nodes_df, "toPandas"):
            nodes_pd = nodes_df.toPandas()
        else:
            nodes_pd = nodes_df.to_pandas()
            
        if hasattr(edges_df, "toPandas"):
            edges_pd = edges_df.toPandas()
        else:
            edges_pd = edges_df.to_pandas()
            
        # Add label column for better visualization
        nodes_pd['label'] = nodes_pd['id']
        # Add fixed node size column
        nodes_pd['size'] = 10
        
        # Create Graphistry graph with minimal bindings
        g = graphistry.edges(edges_pd, 'src', 'dst').nodes(nodes_pd, 'id')
        
        # Add bindings one at a time with error handling
        try:
            g = g.bind(point_title='id')
        except Exception as e:
            print(f"Info: Could not bind point_title: {e}")
            
        try:
            if 'subtype' in nodes_pd.columns:
                g = g.bind(point_color='subtype')
        except Exception as e:
            print(f"Info: Could not bind point_color: {e}")
            
        try:
            g = g.bind(edge_label='relationship')
        except Exception as e:
            print(f"Info: Could not bind edge_label: {e}")
        
        # Apply only basic settings that should be supported in most versions
        try:
            g = g.settings(url_params={'strongGravity': 'true'})
        except Exception as e:
            print(f"Info: Could not apply basic settings: {e}")
        
        # Try to get a URL first (more reliable approach)
        try:
            plot_url = g.plot(render=False)
            print(f"Visualization URL (open in browser): {plot_url}")
            
            # Display the URL as a clickable link
            from IPython.display import display, HTML
            display(HTML(f"<a href='{plot_url}' target='_blank'>Open Graphistry Visualization</a>"))
            
            # Try to embed as iframe if allowed
            try:
                iframe_html = f"""
                <iframe src="{plot_url}" 
                        width="100%" height="500px" 
                        style="border: none; max-width: 100%;">
                </iframe>
                """
                display(HTML(iframe_html))
            except Exception as iframe_err:
                print(f"Info: Could not embed iframe: {iframe_err}")
                
        except Exception as url_err:
            print(f"Warning: Could not generate URL: {url_err}")
            # Fall back to direct rendering
            try:
                plot = g.plot()
                print(" Graphistry visualization created directly")
                from IPython.display import display, HTML
                display(HTML(plot))
            except Exception as plot_err:
                print(f"Error: Could not render plot directly: {plot_err}")
                return None
        
        print(" Graphistry visualization created")
        return g
    except Exception as e:
        print(f"Visualization error: {e}")
        import traceback
        traceback.print_exc()
        return None

# COMMAND ----------

# MAGIC %md
# MAGIC # 7. Save to Delta Table -- setup config

# COMMAND ----------

# --- Save to Delta Function ---
def save_graph_to_delta(nodes_df, edges_df):
    """Save graph structure to Delta tables."""
    extracted_nodes_table_name = "llm_sandbox.aiphs.extracted_kg_nodes_limited_run" 
    extracted_edges_table_name = "llm_sandbox.aiphs.extracted_kg_edges_limited_run" 
    try:
        print(f"Saving nodes to Delta table: {extracted_nodes_table_name}")
        nodes_df.write.format("delta").mode("overwrite").saveAsTable(extracted_nodes_table_name)
        print(f"Saving edges to Delta table: {extracted_edges_table_name}")
        edges_df.write.format("delta").mode("overwrite").saveAsTable(extracted_edges_table_name)
        print(f" Graph saved to Delta tables successfully")
    except Exception as e:
        print(f"Could not save to Delta tables. Error: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC # 8. Setup Testing Functions
# MAGIC * We can test single, scaling, and run a pipeline. We will set this up below. 
# MAGIC
# MAGIC ## 8a. Setup for Test with Single Patient Record

# COMMAND ----------

# --- Process Single Patient Record ---
def test_single_record():
    """Process a single patient record and visualize with Graphistry."""
    print(" TESTING WITH SINGLE PATIENT RECORD")
    print("=" * 60)
    
    start_time = time.time()
    
    # Load a single patient record
    patient_data_df = load_and_process_patient_data(1)
    
    if patient_data_df is None or len(patient_data_df) == 0:
        print(" No data loaded for testing")
        return None
    
    print(f" Patient data loaded successfully")
    
    # Build graph for this patient
    result = build_patient_graph(patient_data_df, visualize=True, save_to_delta=False)
    
    if result:
        kg_graph, nodes_df, edges_df, timing_stats = result
        total_time = time.time() - start_time
        
        print(f"\n SINGLE RECORD PROCESSING COMPLETE")
        print(f"⏱  Total time for 1 patient record: {total_time:.2f} seconds")
        print(f" Graph contains {timing_stats['num_vertices']} vertices and {timing_stats['num_edges']} edges")
        
        return result
    else:
        print(" Graph building failed")
        return None

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8b. Setup for Scaling Testing
# MAGIC * This would allow you to compare building multiple graphs at the same time.
# MAGIC * For now, I have commented this out, but we will save this code for future usage if needed.

# COMMAND ----------

# # --- Scale Testing Function ---
# def scale_test(record_counts=[1, 5, 10]):
#     """Test graph building with different numbers of records."""
#     print(" SCALING TEST")
#     print("=" * 60)
#     results = {}
#     for count in record_counts:
#         print(f"\n Testing with {count} record(s)...")
#         start_time = time.time()
#         try:
#             patient_data_df = load_and_process_patient_data(count)
#             if patient_data_df is None:
#                 print(f" Failed to load {count} records")
#                 continue
#             kg_graph, nodes_df, edges_df, timing_stats = build_patient_graph(
#                 patient_data_df, 
#                 visualize=False,  # Skip visualization for scaling tests
#                 save_to_delta=False
#             )
#             total_time = time.time() - start_time
#             results[count] = {
#                 'total_time': total_time,
#                 'vertices': timing_stats['num_vertices'],
#                 'edges': timing_stats['num_edges'],
#                 'time_per_record': total_time / count
#             }
#             print(f" {count} records: {total_time:.2f}s ({total_time/count:.2f}s per record)")
#         except Exception as e:
#             print(f" Error with {count} records: {e}")
#             results[count] = None
#     print(f"\n SCALING TEST SUMMARY")
#     print(f"{'Records':<10}{'Total Time':<12}{'Per Record':<12}{'Vertices':<10}{'Edges':<8}")
#     print("-" * 60)
#     for count, result in results.items():
#         if result:
#             print(f"{count:<10}{result['total_time']:<12.2f}{result['time_per_record']:<12.2f}{result['vertices']:<10}{result['edges']:<8}")
#         else:
#             print(f"{count:<10}{'FAILED':<12}{'N/A':<12}{'N/A':<12}{'N/A':<8}")
#     return results

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8c. Batch testing -- (e.g. 1000 records at a time)
# MAGIC * Allows you to build a large number of graphs at a time. 
# MAGIC * This is important to build and analyze because the registry number is ~7,000 and growing.

# COMMAND ----------

# --- Process Multiple Records and Build Combined Graph ---
def process_large_batch(num_records=1000, visualize=True, save_to_delta=False):
    """Process multiple patient records and build a single combined graph."""
    print(f" PROCESSING {num_records} RECORDS IN BATCH")
    print("=" * 60)
    
    start_time = time.time()
    
    # Load multiple patient records
    print(f"Loading {num_records} patient records from Delta table...")
    patient_data_df = load_and_process_patient_data(num_records)
    
    if patient_data_df is None or len(patient_data_df) == 0:
        print(" No data loaded. Check your Delta table connection.")
        return None
        
    actual_records = len(patient_data_df)
    if actual_records < num_records:
        print(f" Requested {num_records} records but only found {actual_records}")
    
    print(f"Processing {actual_records} records as a single batch...")
    
    # Process all records at once to build a combined graph
    result = build_patient_graph(patient_data_df, visualize=visualize, save_to_delta=save_to_delta)
    
    if not result:
        print(" Failed to build graph")
        return None
        
    kg_graph, nodes_df, edges_df, timing_stats = result
    
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"BATCH PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Records processed: {actual_records}")
    print(f"Graph contains {timing_stats['num_vertices']} vertices and {timing_stats['num_edges']} edges")
    print(f"Total processing time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"Average time per record: {total_time/actual_records:.2f} seconds")
    
    return kg_graph, nodes_df, edges_df, timing_stats

# COMMAND ----------

# MAGIC %md
# MAGIC # 9. Setup Main Function

# COMMAND ----------

# --- Main Execution ---
def main():
    """Main function to run the graph building process."""
    print("Starting Patient Records Graph Builder with Polars")
    print("=" * 60)
    NUM_RECORDS_TO_PROCESS = 1  # Start with 1 record for testing
    ENABLE_VISUALIZATION = True
    SAVE_TO_DELTA = False
    try:
        print(f"Loading {NUM_RECORDS_TO_PROCESS} patient record(s)...")
        patient_data_df = load_and_process_patient_data(NUM_RECORDS_TO_PROCESS)
        if patient_data_df is None:
            print(" Failed to load patient data. Exiting.")
            return
        kg_graph, nodes_df, edges_df, timing_stats = build_patient_graph(
            patient_data_df, 
            visualize=ENABLE_VISUALIZATION, 
            save_to_delta=SAVE_TO_DELTA
        )
        print(f"\n{'='*60}")
        print(f"FINAL SUMMARY")
        print(f"{'='*60}")
        print(f"Records processed: {NUM_RECORDS_TO_PROCESS}")
        print(f"Graph vertices: {timing_stats['num_vertices']}")
        print(f"Graph edges: {timing_stats['num_edges']}")
        print(f"Total processing time: {timing_stats['total_time']:.2f} seconds")
        print(f"Average time per record: {timing_stats['total_time']/NUM_RECORDS_TO_PROCESS:.2f} seconds")
        if NUM_RECORDS_TO_PROCESS == 1:
            print(f"\n⏱  TIME TO BUILD 1 PATIENT RECORD GRAPH: {timing_stats['total_time']:.2f} seconds")
        if NUM_RECORDS_TO_PROCESS < 100:
            estimated_time_100 = (timing_stats['total_time'] / NUM_RECORDS_TO_PROCESS) * 100
            estimated_time_1000 = (timing_stats['total_time'] / NUM_RECORDS_TO_PROCESS) * 1000
            print(f"\n SCALING ESTIMATES:")
            print(f"   100 records: ~{estimated_time_100/60:.1f} minutes")
            print(f"   1000 records: ~{estimated_time_1000/60:.1f} minutes")
        print(f"\n Graph building completed successfully!")
        return kg_graph, nodes_df, edges_df, timing_stats
    except Exception as e:
        print(f" Error in main execution: {e}")
        import traceback
        traceback.print_exc()
        return None

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC # 10. Run program
# MAGIC * This will setup a simple Databricks widget interface and run processing on `n` number of records that you choose.

# COMMAND ----------

# --- Simple Databricks Widgets ---
dbutils.widgets.dropdown("processing_mode", "single", ["single", "batch"])
dbutils.widgets.text("record_count", "10")

# COMMAND ----------

# --- Run Selected Processing Mode ---
if __name__ == "__main__":
    # Get widget values
    processing_mode = dbutils.widgets.get("processing_mode")
    record_count = int(dbutils.widgets.get("record_count"))
    
    try:
        if processing_mode == "batch":
            print(f" Starting batch processing with {record_count} records...")
            batch_result = process_large_batch(
                num_records=record_count, 
                visualize=True,
                save_to_delta=False
            )
            print("\n Batch processing completed!")
        else:  # single mode
            print(" Starting single record processing...")
            single_result = test_single_record()
            print("\n Single record processing completed!")
    except Exception as e:
        import traceback
        print(f" Error in processing: {e}")
        traceback.print_exc()
