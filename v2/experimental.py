import io
import os
from typing import Union, Optional, Any, Dict, List
import networkx as nx
from openai import OpenAI
import matplotlib.pyplot as plt
from pydantic import BaseModel, Field, validator, field_validator, model_validator
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from docx import Document
from pypdf import PdfReader

app = FastAPI()
graph = nx.Graph()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class Node(BaseModel):
    uid: int
    label: str
    ontology_type: str

    def to_networkx_node(self, g):
        g.add_node(
            self.uid,
            label=self.label,
            type=self.ontology_type)

class Edge(BaseModel):
    source_node_uid: int
    target_node_uid: int
    semantic_relation: str
    edge_weight: float

    def to_networkx_edge(self, g):
        return g.add_edge(
            self.source_node_uid,
            self.target_node_uid,
            weight=self.edge_weight,
            relation=self.semantic_relation)

class GraphPayload(BaseModel):
    nodes: List[Node]
    edges: List[Edge]

async def extract_text(upload: UploadFile) -> str:
    filename = upload.filename or ""
    suffix = filename.lower().rsplit(".", 1)[-1]
    raw: bytes = await upload.read()
    if suffix == "pdf":
        reader = PdfReader(io.BytesIO(raw))
        text_chunks: List[str] = [
            page.extract_text() or "" for page in reader.pages
        ]
        return "\n".join(text_chunks)
    elif suffix.lower() == "docx":
        doc = Document(io.BytesIO(raw))
        # Gather each paragraph’s text, preserving soft line breaks
        return "\n".join(p.text for p in doc.paragraphs)
    else:
        return raw.decode("utf-8", errors="ignore")

@app.post("/ingest_doc")
async def ingest(file: UploadFile = File(...)):
    text = await extract_text(file)
    response = client.beta.chat.completions.parse(
        model="o4-mini",
        messages=[{
            "role": "user",
            "content": "turn the supplied document into a graph of entities (nodes) and relationships (edges)"
                       f"\n---\n "
                       f"{text}"
        }],
        response_format=GraphPayload
    )
    graph_response = response.choices[0].message.parsed

    for node in graph_response.nodes:
        node.to_networkx_node(graph)
    for edge in graph_response.edges:
        edge.to_networkx_edge(graph)


@app.get("/graph", response_class=StreamingResponse)
async def get_graph():
    if graph.number_of_nodes() == 0:
        raise HTTPException(status_code=404, detail="Knowledge graph is empty")

    fig, ax = plt.subplots(figsize=(18, 16))
    pos = nx.spring_layout(graph, seed=42)
    colors = ["lightgreen" if graph.nodes[n].get("type") == "chunk" else "skyblue" for n in graph]
    nx.draw_networkx(graph, pos, node_color=colors, with_labels=True, node_size=300, font_size=7)
    node_labels = {
        n: graph.nodes[n].get("label", str(n))  # fall back to uid if no label
        for n in graph.nodes
    }
    nx.draw_networkx_labels(graph, pos, labels=node_labels, font_size=7, ax=ax)
    edge_labels = {
        (u, v): data.get("relation", "")  # semantic_relation attr
        for u, v, data in graph.edges(data=True)
    }
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels,
                                 font_size=6, ax=ax, label_pos=0.5)
    ax.axis("off")
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png", headers={"Cache-Control": "no-store"})
