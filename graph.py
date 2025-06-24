import json
import os
from pathlib import Path
from tqdm import tqdm
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
import networkx as nx
from langchain_text_splitters import CharacterTextSplitter
from ollama import chat
from pydantic import BaseModel
from matplotlib import pyplot as plt
from prompts import generate_entity_types, normalise_entity_types, generate_entity_relationships, focus_document

class FocusedDocument(BaseModel):
    document_content: str

class EntityTypes(BaseModel):
    entity_types: list[str]

class Entity(BaseModel):  # need to assign a unique id to each, then when we do graph.neighbours(node) we can look up the exact entity for each neighbor
    entity_name: str
    entity_type: str
    entity_description: str

class Relationship(BaseModel):
    source_entity: str
    target_entity: str
    relationship_description: str
    relationship_strength: int

class RelationshipCollection(BaseModel):
    relationships: list[Relationship]

class EntityRelationships(BaseModel):
    entities: list[Entity]
    relationships: list[Relationship]

# --- Configuration ---
chunk_size = 800
overlap = 100
llm_model_name = "qwen3:32b"
input_doc_name = "Technical Description.docx"

chunks = []
entity_types = []
entities = []
relationships = []

input_json = Path(f"{input_doc_name}.json")
if input_json.exists():
    with open(input_json, "rb") as fr:
        er = EntityRelationships.model_validate(json.load(fr))
        entities = er.entities
        relationships = er.relationships
else:
    # --- Document Loader ---
    unstructured_text = ""
    file_path = os.path.join(f"input/{input_doc_name}")
    with open(file_path, "rb") as f:
        file = f.read()

        documents = []
        if input_doc_name.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif input_doc_name.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        elif input_doc_name.endswith(".txt"):
            loader = TextLoader(file_path)

        documents.extend(loader.load())

        combined_doc = " ".join([doc.page_content for doc in documents])

        token_splitter = CharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base", chunk_size=1000, chunk_overlap=100
        )
        texts = token_splitter.split_text(combined_doc)
        focused_docs = []

        print("Focusing document")
        for txt in tqdm(texts):
            try:
                focus_prompt = focus_document(txt)
                focus_res = chat(
                    model=llm_model_name,
                    messages=[{"role": "user", "content": focus_prompt}],
                    format=FocusedDocument.model_json_schema()
                )
                focused_docs.append(FocusedDocument.model_validate_json(focus_res.message.content).document_content)
            except Exception as e:
                print(f"Focus failed: {e}")
                exit()

        focused_doc = " ".join([focus for focus in focused_docs])

        # --- Text splitting ---
        text_splitter = CharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separator="\n"
        )

        print("Splitting document into chunks")
        chunks = text_splitter.split_text(focused_doc)

        # --- Entity Type Extraction ---
        print("Extracting entity types")
        for chunk in tqdm(chunks):
            try:
                entity_types_prompt = generate_entity_types(chunk)
                types_res = chat(
                    model=llm_model_name,
                    messages=[{"role": "user", "content": entity_types_prompt}],
                    format=EntityTypes.model_json_schema(),
                )
                entity_types.extend(EntityTypes.model_validate_json(types_res.message.content).entity_types)
            except Exception as e:
                print(f"EntityTypes extraction failed: {e}")
                exit()
        print(f"Identified entity types ({len(entity_types)}): {[et for et in entity_types]}")

        # --- Entity Type Normalisation ---
        print("Normalising entity types")
        try:
            normalise_prompt = normalise_entity_types(entity_types)
            normalise_res = chat(
                model=llm_model_name,
                messages=[{"role": "user", "content": normalise_prompt}],
                format=EntityTypes.model_json_schema(),
            )
            normalised_entity_types = EntityTypes.model_validate_json(normalise_res.message.content)
            entity_types = [t.lower() for t in normalised_entity_types.entity_types]
            de_duped_types = []
            for et in entity_types:
                if et not in de_duped_types:
                    de_duped_types.append(et)
            entity_types = de_duped_types
        except Exception as e:
            print(f"EntityTypes extraction failed: {e}")
            exit()
        print(f"Normalised entity types ({len(entity_types)}: {[et for et in entity_types]}")

        # --- Entity & Relationship Extraction ---
        print("Identifying entities & relationships")
        idx = 0
        for chunk in tqdm(chunks):
            try:
                generate_prompt = generate_entity_relationships(chunk, entity_types)
                res = chat(
                    model=llm_model_name,
                    messages=[{"role": "user", "content": generate_prompt}],
                    format=EntityRelationships.model_json_schema(),
                )
                entity_relationships = EntityRelationships.model_validate_json(res.message.content)
                entities.extend(entity_relationships.entities)
                if len(entity_relationships.relationships) > 0:
                    relationships.extend(entity_relationships.relationships)
            except Exception as e:
                print(f"EntityTypes extraction failed; {e}")
                exit()
        print(f"Identified entities ({len(entities)}): {[e.entity_name for e in entities]}")
        print(f"Identified relationships ({len(relationships)}): {[f'{r.source_entity} -> {r.target_entity}' for r in relationships]}")

        # # --- Entity Relationship Normalisation ---
        # print("Normalising entity relationships")
        # try:
        #     relationship_normalise_prompt = normalise_relationships(json.dumps([r.dict() for r in relationships], indent=2))
        #     relation_norm_res = chat(
        #         model=llm_model_name,
        #         messages=[{"role": "user", "content": relationship_normalise_prompt}],
        #         format=RelationshipCollection.model_json_schema(),
        #     )
        #     relationships = RelationshipCollection.model_validate_json(relation_norm_res.message.content).relationships
        # except Exception as e:
        #     print(f"Relationship normalisation failed: {e}")
        #     exit()
        # print(f"Normalised relationships ({len(relationships)}): {[f'{r.source_entity} -> {r.target_entity}' for r in relationships]}")

        # --- Summary ---
        print(f"\n--- Overall Summary ---\n")
        print(f"Total chunks defined: {len(chunks)}\")\n")
        print(f"Entity Types defined: {len(entity_types)}\")\n")
        print(f"Total entities extracted: {len(entities)}\")\n")
        print(f"Total relationships discovered: {len(relationships)}\")\n")

        final_entity_relationships = EntityRelationships(
            entities=entities,
            relationships=relationships
        )

        with open(f"{input_doc_name}.json", "w", encoding="utf-8") as fs:
            json.dump(final_entity_relationships.model_dump(), fs, indent=2)

# Create an empty directed graph
knowledge_graph = nx.MultiDiGraph()

print("Adding entity relationships to the knowledge graph")
added_edges_count = 0
for relationship in tqdm(relationships):
    knowledge_graph.add_edge(
        relationship.source_entity,
        relationship.target_entity,
        label=relationship.relationship_description,
        size=relationship.relationship_strength
    )
    added_edges_count += 1

# --- Final Graph Statistics ---
num_nodes = knowledge_graph.number_of_nodes()
num_edges = knowledge_graph.number_of_edges()

print(f"\n--- Final NetworkX Graph Summary ---\n")
print(f"Total unique nodes (entities): {num_nodes}")
print(f"Total unique edges (relationships): {num_edges}")

if num_edges != added_edges_count and isinstance(knowledge_graph, nx.MultiDiGraph):
    print(
        f"Note: Added {added_edges_count} edges, but graph has {num_edges}. DiGraph overwrites edges with same source/target. Use MultiDiGraph if multiple edges needed.")

if num_nodes > 0:
    try:
        density = nx.density(knowledge_graph)
        print(f"Graph density: {density:.4f}")  # How connected the graph is
        if nx.is_weakly_connected(knowledge_graph):  # Can you reach any node from any other, ignoring edge direction?
            print("The graph is weakly connected (all nodes reachable ignoring direction).")
        else:
            num_components = nx.number_weakly_connected_components(knowledge_graph)
            print(f"The graph has {num_components} weakly connected components.")
    except Exception as e:
        print(f"Could not calculate some graph metrics: {e}")  # Handle potential errors on empty/small graphs
else:
    print("Graph is empty, cannot calculate metrics.")
print("-" * 25)

# mock search
query = "analytics server"
matched_nodes = list(set([node for node in knowledge_graph.nodes if any(word in node.lower() for word in query)])) # unique only
related_nodes = []
for node in matched_nodes:
    related_nodes.extend(list(knowledge_graph.neighbors(node)))

neighbor_entities = [entity for entity in entities if any(entity.entity_name.lower() == neighbor.lower() for neighbor in related_nodes)]
unrelated_types = ['permission', 'affiliation', 'legal entity', 'trademark', 'product version', 'section', 'document content', 'document section', 'document', 'intellectual property right', 'legal entity', 'license feature', 'hardware specification', 'appendix', 'hardware specification', 'document', 'document section']
unique_neighbor_entity_types = list(set([nent.entity_type for nent in neighbor_entities if nent.entity_type not in unrelated_types]))
related_entities = [nent for nent in neighbor_entities if nent.entity_type.lower() not in unrelated_types and 'document' not in nent.entity_description and 'section' not in nent.entity_description and 'Table' not in nent.entity_description]

neighbor_relationships = [relation for relation in relationships if
                          any(neighbor.entity_name.lower() == relation.source_entity.lower() for neighbor in related_entities) or
                          any(neighbor.entity_name.lower() == relation.target_entity.lower() for neighbor in related_entities)]

close_neighbor_relations = [close for close in neighbor_relationships if close.relationship_strength > 8]




# Compute layout once
pos = nx.spring_layout(knowledge_graph, seed=0)

# Draw graph with consistent layout
nx.draw(knowledge_graph, pos, with_labels=True, node_color='lightblue', node_size=1500)

# Get edge labels
edge_labels = nx.get_edge_attributes(knowledge_graph, 'label')

# Draw edge labels using the same layout
nx.draw_networkx_edge_labels(knowledge_graph, pos, edge_labels=edge_labels, font_color='red')

plt.show()
