from __future__ import annotations

import sys
print("=== CitationHub app.py starting ===", flush=True)

import base64
import os
from pathlib import Path
from typing import List

import pandas as pd
import networkx as nx
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pyvis.network import Network
import streamlit.components.v1 as components

HF_REPO_ID = os.environ.get("HF_REPO_ID", "")

def csv_download_link(data: bytes, filename: str, label: str) -> None:
    """st.download_button 대신 base64 HTML 링크로 다운로드 — 서버 연결 불필요."""
    b64 = base64.b64encode(data).decode()
    st.markdown(
        f'<a href="data:text/csv;base64,{b64}" download="{filename}" '
        f'style="display:block;text-align:center;padding:8px 12px;'
        f'background:#1e293b;color:white;border-radius:8px;'
        f'text-decoration:none;font-size:14px;width:100%;box-sizing:border-box;">'
        f'{label}</a>',
        unsafe_allow_html=True,
    )
HF_TOKEN   = os.environ.get("HF_TOKEN", "")

st.set_page_config(page_title="CitationHub", page_icon="📚", layout="wide")
print("=== set_page_config done ===", flush=True)

ALLOWED_INTENTS = [
    "background","uses","similarities","motivation",
    "differences","future_work","extends",
]
INTENT_COLORS = {
    "background":"#94a3b8","uses":"#22c55e","similarities":"#3b82f6",
    "motivation":"#f59e0b","differences":"#ef4444",
    "future_work":"#8b5cf6","extends":"#06b6d4",
}
NODE_COLORS = {
    "seed_paper":"#111827","citing_paper":"#dbeafe","citation_event":"#fde68a",
    "journal":"#ede9fe","author":"#fee2e2","affiliation":"#fae8ff",
    "city":"#cffafe","country":"#ffedd5","field":"#e0e7ff","intent":"#dcfce7",
}
NODE_TYPE_COLORS = {
    "seed_paper":"#111827","citing_paper":"#3b82f6","citation_event":"#f59e0b",
    "journal":"#8b5cf6","author":"#ef4444","affiliation":"#ec4899",
    "city":"#06b6d4","country":"#f97316","field":"#6366f1","intent":"#22c55e",
}

DEFAULT_DATA_DIR = Path(os.environ.get(
    "CITATIONHUB_DATA_DIR",
    r"C:\Users\user\OneDrive\바탕 화면\Citehub_huggingface\data",
))

def fmt_num(x):
    try: return f"{int(x):,}"
    except: return "-"

def _hf_download(filename: str) -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download(
        repo_id=HF_REPO_ID, repo_type="dataset",
        filename=f"data/{filename}", token=HF_TOKEN or None,
    )

def _read(filename: str, data_dir: Path | None = None) -> pd.DataFrame:
    if HF_REPO_ID:
        return pd.read_parquet(_hf_download(filename))
    return pd.read_parquet(data_dir / filename)

def plotly_network_fig(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    title: str = "",
    height: int = 750,
    seed_node_ids: list | None = None,
) -> go.Figure:
    """SVG 기반 Plotly 네트워크 그래프 — 확대해도 선명."""
    G = nx.Graph()
    node_meta: dict = {}
    for _, row in nodes_df.iterrows():
        nid = str(row["node_id"])
        G.add_node(nid)
        node_meta[nid] = row

    for _, row in edges_df.iterrows():
        s, t = str(row["source"]), str(row["target"])
        if s in node_meta and t in node_meta:
            G.add_edge(s, t, edge_type=row.get("edge_type", ""))

    if len(G.nodes) == 0:
        return go.Figure()

    k = max(1.5, 3.0 / (len(G.nodes) ** 0.4))
    pos = nx.spring_layout(G, seed=42, k=k, iterations=60)

    ex, ey = [], []
    for src, tgt in G.edges():
        x0, y0 = pos.get(src, (0, 0))
        x1, y1 = pos.get(tgt, (0, 0))
        ex += [x0, x1, None]
        ey += [y0, y1, None]

    traces: list[go.BaseTraceType] = [
        go.Scatter(
            x=ex, y=ey, mode="lines",
            line=dict(width=0.8, color="#cbd5e1"),
            hoverinfo="none", showlegend=False,
        )
    ]

    for ntype, color in NODE_TYPE_COLORS.items():
        subset = nodes_df[nodes_df["node_type"] == ntype]
        if subset.empty:
            continue
        xs, ys, hovers, texts = [], [], [], []
        for _, row in subset.iterrows():
            nid = str(row["node_id"])
            if nid not in pos:
                continue
            x, y = pos[nid]
            xs.append(x); ys.append(y)
            label = str(row.get("label", ""))[:50]
            texts.append(label if ntype == "seed_paper" else "")
            hovers.append(
                f"<b>{label}</b><br>"
                f"Type: {ntype}<br>"
                f"DOI: {row.get('doi','') or '-'}<br>"
                f"Pub: {row.get('publication_name','') or '-'}<br>"
                f"Group: {row.get('group','') or '-'}"
            )

        is_seed = ntype == "seed_paper"
        traces.append(go.Scatter(
            x=xs, y=ys,
            mode="markers+text" if is_seed else "markers",
            text=texts, textposition="top center",
            hovertext=hovers, hoverinfo="text",
            name=ntype,
            marker=dict(
                size=20 if is_seed else 10,
                color=color,
                line=dict(width=1.5 if is_seed else 0.5, color="white"),
                symbol="circle",
            ),
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        showlegend=True,
        legend=dict(title="Node type", itemsizing="constant"),
        hovermode="closest",
        height=height,
        margin=dict(l=0, r=0, t=40 if title else 10, b=0),
        paper_bgcolor="white",
        plot_bgcolor="#f8fafc",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )
    return fig

def plotly_ontology_fig(height: int = 820) -> go.Figure:
    """CitationHub 온톨로지 구조 — Plotly SVG. 각 노드에 속성값 표시."""

    NODE_PROPS = {
        "seed_paper":     "doi · title · journal\nauthor · affiliation\ncountry · field · citedby_count",
        "citation_event": "event_id · citing_year\nprimary_intent · context\nis_influential",
        "citing_paper":   "doi · title\nyear · venue · oa_pdf",
        "intent":         "background · uses\nsimilarities · motivation\ndifferences · future_work · extends",
        "journal":        "journal_name",
        "author":         "author_name · author_id",
        "affiliation":    "affiliation_name",
        "city":           "city_name",
        "country":        "country_name",
        "field":          "field_name",
    }

    node_defs = [
        ("seed",        "Top5PctCitedPaper", "seed_paper"),
        ("event",       "CitationEvent",     "citation_event"),
        ("citing",      "CitingPaper",        "citing_paper"),
        ("intent",      "Intent",             "intent"),
        ("journal",     "Journal",            "journal"),
        ("author",      "Author",             "author"),
        ("affiliation", "Affiliation",        "affiliation"),
        ("city",        "City",               "city"),
        ("country",     "Country",            "country"),
        ("field",       "Field",              "field"),
    ]
    edge_defs = [
        ("event","citing","hasCitingPaper"),    ("event","seed","hasCitedPaper"),
        ("event","intent","hasPrimaryIntent"),   ("seed","journal","publishedInJournal"),
        ("seed","author","hasAuthor"),           ("seed","affiliation","hasAffiliation"),
        ("seed","city","locatedInCity"),         ("seed","country","locatedInCountry"),
        ("seed","field","belongsToField"),
    ]
    G = nx.DiGraph()
    for nid, _, _ in node_defs:
        G.add_node(nid)
    for s, t, _ in edge_defs:
        G.add_edge(s, t)

    pos = nx.spring_layout(G, seed=7, k=2.5, iterations=80)

    ex, ey = [], []
    ann = []
    for s, t, lbl in edge_defs:
        x0, y0 = pos[s]; x1, y1 = pos[t]
        ex += [x0, x1, None]; ey += [y0, y1, None]
        mx, my = (x0+x1)/2, (y0+y1)/2
        ann.append(dict(
            x=mx, y=my, text=f"<i>{lbl}</i>",
            showarrow=False, font=dict(size=9, color="#64748b"),
            bgcolor="rgba(255,255,255,0.75)",
        ))

    traces: list[go.BaseTraceType] = [
        go.Scatter(x=ex, y=ey, mode="lines",
                   line=dict(width=1.2, color="#94a3b8"),
                   hoverinfo="none", showlegend=False)
    ]

    for nid, label, ntype in node_defs:
        x, y = pos[nid]
        color = NODE_TYPE_COLORS.get(ntype, "#94a3b8")
        props = NODE_PROPS.get(ntype, "")

        traces.append(go.Scatter(
            x=[x], y=[y], mode="markers+text",
            text=[f"<b>{label}</b>"], textposition="top center",
            hoverinfo="text",
            hovertext=(f"<b>{label}</b><br>Type: {ntype}<br>"
                       + props.replace("\n", "<br>")),
            name=label, showlegend=False,
            marker=dict(size=24, color=color,
                        line=dict(width=1.5, color="white")),
            textfont=dict(size=11, color="#1e293b"),
        ))

        if props:
            prop_html = props.replace("\n", "<br>")
            ann.append(dict(
                x=x, y=y,
                text=f"<span style='font-size:8px;color:#64748b'>{prop_html}</span>",
                showarrow=False,
                xanchor="center",
                yanchor="top",
                yshift=-22,
                font=dict(size=8, color="#64748b"),
                bgcolor="rgba(248,250,252,0.85)",
                borderpad=2,
            ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        showlegend=False, hovermode="closest", height=height,
        annotations=ann,
        margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="white", plot_bgcolor="#f8fafc",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )
    return fig

def inject_fullscreen(html: str) -> str:
    extra = """
    <button onclick="var el=document.getElementById('mynetwork');
      if(el){if(el.requestFullscreen)el.requestFullscreen();
      else if(el.webkitRequestFullscreen)el.webkitRequestFullscreen();}"
      style="position:fixed;bottom:18px;right:18px;z-index:9999;
             padding:8px 18px;background:#1e293b;color:white;border:none;
             border-radius:8px;cursor:pointer;font-size:13px;
             box-shadow:0 2px 8px rgba(0,0,0,0.35);">⛶ Fullscreen</button>
    <div style="position:fixed;bottom:18px;left:18px;z-index:9999;font-size:12px;
                color:#64748b;background:rgba(255,255,255,0.85);
                padding:5px 10px;border-radius:6px;">
      🖱 Scroll: zoom &nbsp;|&nbsp; Drag: pan &nbsp;|&nbsp; Click node: info</div>
    <script>
    // HiDPI 캔버스 해상도 보정 (Canvas 흐림 최소화)
    (function fixDPI() {
      var canvas = document.querySelector('#mynetwork canvas');
      if (!canvas) { setTimeout(fixDPI, 200); return; }
      var dpr = window.devicePixelRatio || 1;
      if (dpr <= 1) return;
      try {
        if (typeof network !== 'undefined') {
          network.canvas.pixelRatio = dpr;
          network.redraw();
        }
      } catch(e) {}
    })();
    </script>
    """
    return html.replace("</body>", extra + "</body>")

@st.cache_data(show_spinner=False)
def load_data(data_dir_str: str):
    d = None if HF_REPO_ID else Path(data_dir_str)

    seed_df   = _read("seed_cited_papers_normalized.parquet", d)
    events_df = _read("citation_events_normalized.parquet", d)
    citing_df = _read("citing_papers_normalized.parquet", d)

    seed = pd.DataFrame({
        "seed_paper_id":  seed_df["seed_paper_id"],
        "doi":            seed_df.get("doi", pd.Series(dtype=str)).fillna(""),
        "title":          seed_df.get("title", pd.Series(dtype=str)).fillna(""),
        "journal":        seed_df.get("publication_name", pd.Series(dtype=str)).fillna(""),
        "author":         seed_df.get("creator", pd.Series(dtype=str)).fillna(""),
        "affiliation":    seed_df.get("affilname", pd.Series(dtype=str)).fillna(""),
        "city":           seed_df.get("affiliation_city", pd.Series(dtype=str)).fillna(""),
        "country":        seed_df.get("affiliation_country", pd.Series(dtype=str)).fillna(""),
        "field":          seed_df.get("group", pd.Series(dtype=str)).fillna(""),
        "cover_date":     seed_df.get("cover_date", pd.Series(dtype=str)).fillna(""),
        "citedby_count":  pd.to_numeric(seed_df.get("citedby_count"), errors="coerce").fillna(0).astype(int),
        "author_id":      seed_df.get("author_id", pd.Series(dtype=object)),
        "affiliation_id": seed_df.get("affiliation_id", pd.Series(dtype=object)),
        "country_id":     seed_df.get("country_id", pd.Series(dtype=object)),
        "field_id":       seed_df.get("field_id", pd.Series(dtype=object)),
        "journal_id":     seed_df.get("journal_id", pd.Series(dtype=object)),
    })
    for col in ["title","doi","journal","field","country"]:
        seed[f"{col}_lc"] = seed[col].astype(str).str.lower()
    seed = seed.sort_values(["citedby_count","title"], ascending=[False,True]).reset_index(drop=True)

    events = pd.DataFrame({
        "citation_event_id": events_df["citation_event_id"],
        "seed_paper_id":     events_df["cited_seed_paper_id"],
        "citing_paper_id":   events_df["citing_paper_id"],
        "citing_title":      events_df.get("citing_title", pd.Series(dtype=str)).fillna(""),
        "citing_doi":        events_df.get("citing_doi", pd.Series(dtype=str)).fillna(""),
        "citing_year":       pd.to_numeric(events_df.get("citing_year"), errors="coerce"),
        "citing_venue":      events_df.get("citing_venue", pd.Series(dtype=str)).fillna(""),
        "primary_intent":    events_df.get("primary_intent", pd.Series(dtype=str)).fillna(""),
        "contexts":          events_df.get("contexts"),
        "context_count":     pd.to_numeric(events_df.get("context_count"), errors="coerce").fillna(0).astype(int),
        "intent_count":      pd.to_numeric(events_df.get("intent_count"), errors="coerce").fillna(0).astype(int),
        "is_influential":    events_df.get("is_influential", pd.Series(dtype=bool)).fillna(False),
        "field_id":          events_df.get("field_id", pd.Series(dtype=object)),
    })
    events = events[events["primary_intent"].isin(ALLOWED_INTENTS)].reset_index(drop=True)

    citing = pd.DataFrame({
        "citing_paper_id": citing_df["citing_paper_id"],
        "doi":    citing_df.get("doi",   pd.Series(dtype=str)).fillna(""),
        "title":  citing_df.get("title", pd.Series(dtype=str)).fillna(""),
        "year":   pd.to_numeric(citing_df.get("year"), errors="coerce"),
        "venue":  citing_df.get("venue", pd.Series(dtype=str)).fillna(""),
        "oa_pdf": citing_df.get("oa_pdf",pd.Series(dtype=str)).fillna(""),
    })

    filters = {
        "fields":    sorted([x for x in seed["field"].dropna().astype(str).unique() if x]),
        "countries": sorted([x for x in seed["country"].dropna().astype(str).unique() if x]),
        "journals":  sorted([x for x in seed["journal"].dropna().astype(str).unique() if x]),
        "intents":   ALLOWED_INTENTS,
        "year_min":  int(events["citing_year"].dropna().min()) if events["citing_year"].notna().any() else 2000,
        "year_max":  int(events["citing_year"].dropna().max()) if events["citing_year"].notna().any() else 2025,
    }
    overview = {
        "seed_papers":     int(len(seed)),
        "citation_events": int(len(events)),
        "citing_papers":   int(events["citing_paper_id"].nunique()),
        "authors":         int(seed["author"].replace("", pd.NA).dropna().nunique()),
        "journals":        int(seed["journal"].replace("", pd.NA).dropna().nunique()),
        "countries":       int(seed["country"].replace("", pd.NA).dropna().nunique()),
        "fields":          int(seed["field"].replace("", pd.NA).dropna().nunique()),
        "intents":         len(ALLOWED_INTENTS),
    }
    return seed, events, citing, filters, overview

@st.cache_data(show_spinner=False)
def load_authors_data(data_dir_str: str) -> pd.DataFrame:
    """Analytics 탭에서만 사용 — 탭 진입 시 로드"""
    d = None if HF_REPO_ID else Path(data_dir_str)
    return _read("authors.parquet", d)

@st.cache_data(show_spinner=False)
def load_geo_data(data_dir_str: str) -> pd.DataFrame:
    """Geographic Map 탭에서만 사용 — 탭 진입 시 로드"""
    d = None if HF_REPO_ID else Path(data_dir_str)
    return _read("affiliation_geo.parquet", d)

@st.cache_data(show_spinner=False)
def load_kg_nodes(data_dir_str: str) -> pd.DataFrame:
    """kg_nodes 전체 로드 (3.4M rows, ~160MB 파일)"""
    d = None if HF_REPO_ID else Path(data_dir_str)
    return _read("kg_nodes.parquet", d)

@st.cache_data(show_spinner=False)
def get_parquet_path(filename: str, data_dir_str: str) -> str:
    """파일 경로 반환 (HF면 로컬 캐시에 다운로드 후 경로 반환)"""
    if HF_REPO_ID:
        return _hf_download(filename)

    return str(Path(data_dir_str) / filename).replace("\\", "/")

@st.cache_data(show_spinner=False)
def query_kg_edges_for_node(node_id: str, kg_edges_path: str, max_edges: int = 80) -> pd.DataFrame:
    """DuckDB: 특정 노드의 엣지만 parquet에서 바로 쿼리 (전체 로드 없음)"""
    import duckdb
    safe_path = kg_edges_path.replace("\\", "/")
    safe_node = node_id.replace("'", "''")
    q = f"""
    SELECT source, target, edge_type
    FROM read_parquet('{safe_path}')
    WHERE source = '{safe_node}' OR target = '{safe_node}'
    LIMIT {int(max_edges)}
    """
    return duckdb.execute(q).df()

@st.cache_data(show_spinner=False)
def query_enriched_stats(enriched_path: str):
    """DuckDB: enriched 전체 로드 없이 집계 통계만 쿼리"""
    import duckdb
    safe_path = enriched_path.replace("\\", "/")

    sem_df = duckdb.execute(f"""
        SELECT has_semantic_evidence, COUNT(*) AS count
        FROM read_parquet('{safe_path}')
        GROUP BY has_semantic_evidence
    """).df()

    field_df = duckdb.execute(f"""
        SELECT field_folder AS field,
               AVG(CAST(has_semantic_evidence AS INTEGER)) AS sem_ratio,
               COUNT(*) AS event_count
        FROM read_parquet('{safe_path}')
        GROUP BY field_folder
        ORDER BY sem_ratio DESC
        LIMIT 20
    """).df()

    return sem_df, field_df

@st.cache_data(show_spinner=False)
def query_explorer_edges(node_id: str, kg_edges_path: str, max_edges: int = 60) -> pd.DataFrame:
    """DuckDB: KG Explorer용 임의 노드 엣지 쿼리"""
    import duckdb
    safe_path = kg_edges_path.replace("\\", "/")
    safe_node = node_id.replace("'", "''")
    q = f"""
    SELECT source, target, edge_type
    FROM read_parquet('{safe_path}')
    WHERE source = '{safe_node}' OR target = '{safe_node}'
    LIMIT {int(max_edges)}
    """
    return duckdb.execute(q).df()

def filter_seed_papers(seed, q, fields, countries, journals):
    df = seed.copy()
    q = (q or "").strip().lower()
    if q:
        df = df[df["title_lc"].str.contains(q, na=False) | df["doi_lc"].str.contains(q, na=False)]
    if fields:    df = df[df["field"].str.lower().isin({x.lower() for x in fields})]
    if countries: df = df[df["country"].str.lower().isin({x.lower() for x in countries})]
    if journals:  df = df[df["journal"].str.lower().isin({x.lower() for x in journals})]
    return df.reset_index(drop=True)

def event_subset(events, seed_paper_id, year_min, year_max):
    df = events[events["seed_paper_id"] == seed_paper_id].copy()
    df = df[df["citing_year"].fillna(-99999) >= year_min]
    df = df[df["citing_year"].fillna(99999) <= year_max]
    return df.reset_index(drop=True)

def build_intent_summary(df):
    counts = df.groupby("primary_intent").size().to_dict()
    return pd.DataFrame({"intent": ALLOWED_INTENTS,
                          "count": [int(counts.get(i,0)) for i in ALLOWED_INTENTS]})

def build_context_rows(df, limit=20):
    rows = []
    df = df.sort_values(["context_count","intent_count","citing_year"],
                        ascending=[False,False,False], na_position="last")
    for _, row in df.iterrows():
        ctx = row["contexts"]
        if isinstance(ctx, list) and ctx:
            for c in ctx[:2]:
                rows.append({"primary_intent": row["primary_intent"],
                             "citing_title": row["citing_title"],
                             "citing_doi": row["citing_doi"],
                             "citing_year": None if pd.isna(row["citing_year"]) else int(row["citing_year"]),
                             "context": c})
        if len(rows) >= limit: break
    return pd.DataFrame(rows[:limit])

def build_citing_table(df, limit=30):
    if df.empty:
        return pd.DataFrame(columns=["citing_title","citing_year","primary_intent","context_count"])
    return (df.sort_values(["context_count","intent_count","citing_year"],
                            ascending=[False,False,False], na_position="last")
            [["citing_paper_id","citing_title","citing_doi","citing_year","primary_intent","context_count"]]
            .drop_duplicates(subset=["citing_paper_id"]).head(limit))

def get_cocited_papers(selected_seed_id, events, seed, top_n=15):
    """선택된 seed paper를 인용한 논문들이 함께 인용한 다른 seed papers"""
    citing_ids = events[events["seed_paper_id"] == selected_seed_id]["citing_paper_id"].unique()
    cocited = (events[events["citing_paper_id"].isin(citing_ids) &
                      (events["seed_paper_id"] != selected_seed_id)]
               .groupby("seed_paper_id").size()
               .reset_index(name="co_citation_count")
               .sort_values("co_citation_count", ascending=False)
               .head(top_n))
    return cocited.merge(seed[["seed_paper_id","title","field","journal","citedby_count"]],
                         on="seed_paper_id", how="left")

def get_kg_subgraph(seed_doi: str, kg_nodes, kg_edges, max_edges=80):
    """선택된 seed paper의 KG 1-hop 서브그래프 반환"""
    node_id = f"seed:{seed_doi}"
    edges = kg_edges[(kg_edges["source"] == node_id) |
                     (kg_edges["target"] == node_id)].head(max_edges)
    if edges.empty:
        return None, None
    all_node_ids = set(edges["source"].tolist()) | set(edges["target"].tolist())
    nodes = kg_nodes[kg_nodes["node_id"].isin(all_node_ids)]
    return nodes, edges

def get_explorer_subgraph(search_node_id: str, kg_nodes, kg_edges, max_edges=60):
    """KG Explorer: 임의 노드 기준 서브그래프"""
    edges = kg_edges[(kg_edges["source"] == search_node_id) |
                     (kg_edges["target"] == search_node_id)].head(max_edges)
    if edges.empty:
        return None, None
    all_ids = set(edges["source"].tolist()) | set(edges["target"].tolist())
    nodes = kg_nodes[kg_nodes["node_id"].isin(all_ids)]
    return nodes, edges

def pyvis_citation_graph(seed_row, events_df):
    net = Network(height="780px", width="100%", bgcolor="#ffffff", font_color="#111827", directed=True)
    sid = seed_row["seed_paper_id"]
    net.add_node(sid, label=seed_row["title"][:60], color="#111827", size=34, shape="dot",
                 font={"color":"white"})
    for _, row in events_df.sort_values(["context_count","intent_count"],
                                         ascending=False).head(40).iterrows():
        cid = row["citing_paper_id"]
        net.add_node(cid, label=(row["citing_title"] or row["citing_doi"] or cid)[:60],
                     color=NODE_COLORS["citing_paper"], size=18, shape="dot")
        ctx = (row["contexts"] or [])[0] if isinstance(row["contexts"], list) and row["contexts"] else ""
        yr  = "" if pd.isna(row["citing_year"]) else int(row["citing_year"])
        net.add_edge(cid, sid, label=row["primary_intent"],
                     color=INTENT_COLORS.get(row["primary_intent"],"#94a3b8"),
                     title=f"Intent: {row['primary_intent']}<br>Year: {yr}<br>{ctx}")
    net.barnes_hut()
    return inject_fullscreen(net.generate_html())

def pyvis_ontology():
    net = Network(height="780px", width="100%", bgcolor="#ffffff", font_color="#111827", directed=True)
    for nid, label, typ in [
        ("seed","Top5PctCitedPaper","seed_paper"),("event","CitationEvent","citation_event"),
        ("citing","CitingPaper","citing_paper"),  ("intent","Intent","intent"),
        ("journal","Journal","journal"),           ("author","Author","author"),
        ("affiliation","Affiliation","affiliation"),("city","City","city"),
        ("country","Country","country"),           ("field","Field","field"),
    ]:
        net.add_node(nid, label=label, color=NODE_COLORS[typ], size=24)
    for s, t, l in [
        ("event","citing","hasCitingPaper"),("event","seed","hasCitedPaper"),
        ("event","intent","hasPrimaryIntent"),("seed","journal","publishedInJournal"),
        ("seed","author","hasAuthor"),        ("seed","affiliation","hasAffiliation"),
        ("seed","city","locatedInCity"),      ("seed","country","locatedInCountry"),
        ("seed","field","belongsToField"),
    ]:
        net.add_edge(s, t, label=l)
    net.barnes_hut()
    return inject_fullscreen(net.generate_html())

def pyvis_from_kg(nodes_df, edges_df, height="780px"):
    """kg_nodes / kg_edges DataFrame으로 pyvis 그래프 생성"""
    net = Network(height=height, width="100%", bgcolor="#ffffff", font_color="#111827", directed=True)
    for _, row in nodes_df.iterrows():
        ntype = row.get("node_type","")
        color = NODE_TYPE_COLORS.get(ntype,"#94a3b8")
        label = str(row.get("label",""))[:55]
        size  = 30 if ntype == "seed_paper" else 16
        font  = {"color":"white"} if ntype == "seed_paper" else {}
        tooltip = f"Type: {ntype}<br>DOI: {row.get('doi','')}<br>Pub: {row.get('publication_name','')}"
        net.add_node(str(row["node_id"]), label=label, color=color,
                     size=size, shape="dot", title=tooltip, font=font)
    for _, row in edges_df.iterrows():
        net.add_edge(str(row["source"]), str(row["target"]),
                     label=row.get("edge_type",""), color="#94a3b8")
    net.barnes_hut()
    return inject_fullscreen(net.generate_html())

print(f"=== HF_REPO_ID={repr(HF_REPO_ID)} HF_TOKEN={'set' if HF_TOKEN else 'EMPTY'} ===", flush=True)

st.title("CitationHub")
print("=== st.title done ===", flush=True)
st.caption("Explore influential papers (top 5% cited), their citation networks, and knowledge graphs.")

_loading_placeholder = st.empty()

with st.sidebar:
    print("=== entered sidebar ===", flush=True)
    st.subheader("Data source")
    if HF_REPO_ID:
        data_dir_val = "hf"
        st.caption(f"Hugging Face: {HF_REPO_ID}")
    else:
        data_dir_val = st.text_input("Parquet directory", str(DEFAULT_DATA_DIR))

    try:
        print(f"=== calling load_data({data_dir_val!r}) ===", flush=True)
        _loading_placeholder.info("⏳ Loading CitationHub data… this may take a moment on first visit.")
        seed, events, citing, filters, overview = load_data(data_dir_val)
        print("=== load_data done ===", flush=True)
        _loading_placeholder.empty()
        st.success("Data loaded")
    except Exception as e:
        _loading_placeholder.empty()
        st.error(str(e)); st.stop()

    st.subheader("Search seed papers")
    q_input = st.text_input("Title or DOI")
    if "q_submit" not in st.session_state: st.session_state["q_submit"] = ""
    if st.button("Search", use_container_width=True):
        st.session_state["q_submit"] = q_input

    fields_sel    = st.multiselect("Field", filters["fields"])
    countries_sel = st.multiselect("Country", filters["countries"])
    journals_sel  = st.multiselect("Journal", filters["journals"][:200])
    y_min = max(2000, filters["year_min"])
    year_min, year_max = st.slider("Citing year", y_min, filters["year_max"], (y_min, filters["year_max"]))

    seed_filtered = filter_seed_papers(seed, st.session_state["q_submit"],
                                       fields_sel, countries_sel, journals_sel)

    st.subheader("Overview counts")
    c1, c2 = st.columns(2)
    c1.metric("Seed papers",     fmt_num(overview["seed_papers"]))
    c2.metric("Citation events", fmt_num(overview["citation_events"]))
    c1.metric("Citing papers",   fmt_num(overview["citing_papers"]))
    c2.metric("Authors",         fmt_num(overview["authors"]))
    c1.metric("Countries",       fmt_num(overview["countries"]))
    c2.metric("Fields",          fmt_num(overview["fields"]))

    options = seed_filtered["seed_paper_id"].tolist()
    if not options:
        st.warning("No seed papers match the current search."); st.stop()
    current     = st.session_state.get("selected_seed_id", options[0])
    default_idx = options.index(current) if current in options else 0
    selected_seed_id = st.selectbox(
        "Seed paper", options, index=default_idx,
        format_func=lambda sid: seed_filtered.loc[
            seed_filtered["seed_paper_id"]==sid, "title"].iloc[0],
    )
    st.session_state["selected_seed_id"] = selected_seed_id

selected_seed  = seed_filtered[seed_filtered["seed_paper_id"]==selected_seed_id].iloc[0]
seed_events    = event_subset(events, selected_seed_id, year_min, year_max)
intent_summary = build_intent_summary(seed_events)
contexts_df    = build_context_rows(seed_events)
citing_table   = build_citing_table(seed_events)

(tab_overview, tab_cnet,
 tab_kg_exp, tab_geo, tab_analytics) = st.tabs([
    "Overview","Citation Network",
    "Knowledge Graph","Geographic Map","Analytics",
])

with tab_overview:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Seed paper detail")
        dc1, dc2 = st.columns(2)
        dc1.metric("Cited by",        fmt_num(selected_seed["citedby_count"]))
        dc2.metric("Citation events", fmt_num(len(seed_events)))
        for label, key in [
            ("Title","title"),("DOI","doi"),("Published","cover_date"),
            ("Journal","journal"),("Author","author"),("Affiliation","affiliation"),
            ("City","city"),("Country","country"),("Field","field"),
        ]:
            st.markdown(f"**{label}**  \n{selected_seed[key] or '-'}")

        st.subheader("Related citing papers")
        st.dataframe(citing_table.rename(columns={
            "citing_title":"Title","citing_year":"Year",
            "primary_intent":"Intent","context_count":"Contexts"}),
            use_container_width=True, hide_index=True)

        st.subheader("Co-cited seed papers")
        st.caption("Other top 5% cited papers that appear together with the selected paper in the same citing works")
        cocited = get_cocited_papers(selected_seed_id, events, seed)
        if cocited.empty:
            st.info("Co-cited papers not found.")
        else:
            st.dataframe(cocited.rename(columns={
                "co_citation_count":"Co-citations","title":"Title",
                "field":"Field","citedby_count":"Cited by"}),
                use_container_width=True, hide_index=True)

    with col2:
        st.subheader("Intent distribution (selected paper)")
        fig = px.bar(intent_summary, x="intent", y="count", color="intent",
                     color_discrete_map=INTENT_COLORS)
        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("CitationHub Intent Distribution")
        all_intents = events.groupby("primary_intent").size().to_dict()
        ai_df = pd.DataFrame({"intent": ALLOWED_INTENTS,
                               "count": [int(all_intents.get(i, 0)) for i in ALLOWED_INTENTS]})
        fig2 = px.bar(ai_df, x="intent", y="count", color="intent",
                      color_discrete_map=INTENT_COLORS)
        fig2.update_layout(showlegend=False, xaxis_title="", yaxis_title="Count")
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("CitationHub Field Distribution")
        fd = (seed_filtered.groupby("field", dropna=False).size()
              .reset_index(name="count").sort_values("count", ascending=False).head(20))
        fd["field"] = fd["field"].replace("","Unknown")
        st.plotly_chart(
            px.bar(fd, x="field", y="count").update_layout(xaxis_title="", yaxis_title="Count"),
            use_container_width=True)

    st.subheader("Citation contexts")
    if contexts_df.empty:
        st.info("No contexts available.")
    else:
        for _, row in contexts_df.iterrows():
            st.markdown(
                f"""<div style="border:1px solid #e2e8f0;border-radius:14px;padding:12px;
                margin-bottom:10px;background:#f8fafc;">
                <div style="display:inline-block;background:{INTENT_COLORS.get(row['primary_intent'],'#64748b')};
                color:white;border-radius:999px;padding:4px 8px;font-size:12px;margin-bottom:6px;">
                {row['primary_intent']}</div>
                <div style="font-size:12px;color:#64748b;margin-bottom:6px;">
                {row['citing_year'] or '-'} · {row['citing_title'] or row['citing_doi']}</div>
                <div>{row['context']}</div></div>""",
                unsafe_allow_html=True)

with tab_cnet:
    st.subheader("Citation Network")
    st.caption("🖱 Scroll: zoom  |  Drag: pan  |  Click node: info  |  ⛶ button: fullscreen")
    if seed_events.empty:
        st.info("No citation network data for this seed paper.")
    else:
        components.html(pyvis_citation_graph(selected_seed, seed_events), height=820, scrolling=True)

with tab_kg_exp:
    st.subheader("Knowledge Graph")

    st.subheader("CitationHub Ontology — Concepts, Instances & Relationships")
    st.caption("🔍 Scroll/pinch: zoom  |  Drag: pan  |  Hover node: details  |  ⛶ (top-right toolbar): fullscreen")
    st.plotly_chart(plotly_ontology_fig(height=820), use_container_width=True)

    st.markdown("---")

    try:
        with st.spinner("Loading..."):
            kg_nodes_exp  = load_kg_nodes(data_dir_val)
            kg_edges_path = get_parquet_path("kg_edges.parquet", data_dir_val)

        import duckdb as _ddb

        nt = kg_nodes_exp["node_type"].value_counts().reset_index()
        nt.columns = ["node_type", "count"]

        et = _ddb.execute(f"""
            SELECT edge_type, COUNT(*) AS count
            FROM read_parquet('{kg_edges_path}')
            GROUP BY edge_type ORDER BY count DESC
        """).df()

        col_a, col_b, col_c, col_d = st.columns([1, 2, 1, 2])
        with col_a:
            st.subheader("Node Types")
            st.dataframe(nt, use_container_width=True, hide_index=True)
        with col_b:
            st.subheader("CitationHub KG Node Distribution")
            nt_fig = px.bar(nt, x="node_type", y="count", color="node_type",
                            color_discrete_map=NODE_TYPE_COLORS)
            nt_fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Count")
            st.plotly_chart(nt_fig, use_container_width=True)
        with col_c:
            st.subheader("Edge Types")
            st.dataframe(et, use_container_width=True, hide_index=True)
        with col_d:
            st.subheader("CitationHub KG Edge Distribution")
            et_fig = px.bar(et, x="edge_type", y="count", color="edge_type")
            et_fig.update_layout(showlegend=False, xaxis_title="",
                                 yaxis_title="Count", xaxis_tickangle=-35)
            st.plotly_chart(et_fig, use_container_width=True)

        st.markdown("---")
        st.subheader("Multi-Node Knowledge Graph")
        st.caption("🖱 Scroll: zoom  |  Drag: pan  |  Click node: info  |  ⛶ button: fullscreen")

        n_seeds = st.slider("Number of seed papers", 3, 15, 6, key="kg_exp_n_seeds")

        EDGES_PER_TYPE = 10

        with st.spinner("Querying graph..."):
            top_seeds = (kg_nodes_exp[kg_nodes_exp["node_type"] == "seed_paper"]
                         .sort_values("citedby_count", ascending=False)
                         .head(n_seeds))
            seed_ids = top_seeds["node_id"].tolist()

            if seed_ids:
                ids_sql = ", ".join(f"'{sid}'" for sid in seed_ids)

                hop1 = _ddb.execute(f"""
                    WITH ranked AS (
                        SELECT source, target, edge_type,
                               ROW_NUMBER() OVER (
                                   PARTITION BY edge_type ORDER BY source
                               ) AS rn
                        FROM read_parquet('{kg_edges_path}')
                        WHERE source IN ({ids_sql}) OR target IN ({ids_sql})
                    )
                    SELECT source, target, edge_type FROM ranked
                    WHERE rn <= {EDGES_PER_TYPE}
                """).df()

                hop1_all_ids = set(hop1["source"].tolist()) | set(hop1["target"].tolist())
                event_node_ids = (
                    kg_nodes_exp[
                        kg_nodes_exp["node_id"].isin(hop1_all_ids) &
                        (kg_nodes_exp["node_type"] == "citation_event")
                    ]["node_id"].tolist()[:40]
                )

                if event_node_ids:
                    ev_sql = ", ".join(f"'{eid}'" for eid in event_node_ids)

                    hop2 = _ddb.execute(f"""
                        WITH ranked AS (
                            SELECT source, target, edge_type,
                                   ROW_NUMBER() OVER (
                                       PARTITION BY edge_type ORDER BY source
                                   ) AS rn
                            FROM read_parquet('{kg_edges_path}')
                            WHERE (source IN ({ev_sql}) OR target IN ({ev_sql}))
                              AND edge_type NOT IN (
                                  SELECT DISTINCT edge_type
                                  FROM read_parquet('{kg_edges_path}')
                                  WHERE source IN ({ids_sql}) OR target IN ({ids_sql})
                              )
                        )
                        SELECT source, target, edge_type FROM ranked
                        WHERE rn <= {EDGES_PER_TYPE}
                    """).df()
                    exp_edges = pd.concat([hop1, hop2]).drop_duplicates(
                        subset=["source", "target", "edge_type"]
                    )
                else:
                    exp_edges = hop1

                all_exp_ids = set(exp_edges["source"].tolist()) | set(exp_edges["target"].tolist())
                exp_nodes = kg_nodes_exp[kg_nodes_exp["node_id"].isin(all_exp_ids)]

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Nodes",      fmt_num(len(exp_nodes)))
                c2.metric("Edges",      fmt_num(len(exp_edges)))
                c3.metric("Node types", fmt_num(exp_nodes["node_type"].nunique()))
                c4.metric("Edge types", fmt_num(exp_edges["edge_type"].nunique()))

                kg_html = pyvis_from_kg(exp_nodes, exp_edges)
                components.html(kg_html, height=860, scrolling=True)

    except Exception as e:
        st.error(str(e))

with tab_geo:
    st.subheader("Geographic Distribution of Seed Papers")
    with st.spinner("Loading geographic data..."):
        aff_geo_df = load_geo_data(data_dir_val)

    country_cnt = (seed_filtered.groupby("country", dropna=False).size()
                   .reset_index(name="count").rename(columns={"country":"country_name"}))
    country_cnt = country_cnt[country_cnt["country_name"].str.strip() != ""]

    if not country_cnt.empty:
        fig_map = px.choropleth(country_cnt, locations="country_name",
                                locationmode="country names", color="count",
                                hover_name="country_name",
                                color_continuous_scale="Blues",
                                title="Seed Papers by Country")
        fig_map.update_layout(geo=dict(showframe=False), height=500)
        st.plotly_chart(fig_map, use_container_width=True)

    st.subheader("Top Cities")
    city_cnt = (seed_filtered.merge(
                    aff_geo_df[["affiliation_name","city_name","country_name"]],
                    left_on="affiliation", right_on="affiliation_name", how="left")
                .groupby(["country_name","city_name"], dropna=False).size()
                .reset_index(name="count").dropna(subset=["country_name"])
                .sort_values("count", ascending=False).head(30))
    if not city_cnt.empty:
        st.plotly_chart(
            px.bar(city_cnt, x="city_name", y="count", color="country_name",
                   title="Top 30 Cities")
            .update_layout(xaxis_title="", yaxis_title="# Seed Papers", xaxis_tickangle=-40),
            use_container_width=True)

    st.subheader("Top Affiliations")
    geo_col1, geo_col2 = st.columns(2)

    with geo_col1:
        aff_cnt = (seed_filtered[seed_filtered["affiliation"].str.strip() != ""]
                   .groupby("affiliation").size()
                   .reset_index(name="count")
                   .sort_values("count", ascending=False).head(20))
        if not aff_cnt.empty:
            st.plotly_chart(
                px.bar(aff_cnt, x="count", y="affiliation", orientation="h",
                       title="Top 20 Affiliations by Seed Papers",
                       labels={"count": "Seed Papers", "affiliation": ""})
                .update_layout(yaxis=dict(autorange="reversed"),
                               xaxis_title="Seed Papers", yaxis_title="", height=520),
                use_container_width=True)

    with geo_col2:
        aff_country = (seed_filtered[
                (seed_filtered["affiliation"].str.strip() != "") &
                (seed_filtered["country"].str.strip() != "")
            ]
            .groupby(["country", "affiliation"]).size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )
        top_affs = aff_country.groupby("affiliation")["count"].sum().nlargest(20).index
        aff_country_top = aff_country[aff_country["affiliation"].isin(top_affs)]
        if not aff_country_top.empty:
            st.plotly_chart(
                px.bar(aff_country_top, x="count", y="affiliation",
                       color="country", orientation="h",
                       title="Top Affiliations by Country",
                       labels={"count": "Seed Papers", "affiliation": "", "country": "Country"})
                .update_layout(yaxis=dict(autorange="reversed"),
                               barmode="stack",
                               xaxis_title="Seed Papers", yaxis_title="",
                               legend_title="Country", height=520),
                use_container_width=True)

with tab_analytics:
    try:
        with st.spinner("Loading analytics data..."):
            authors_df = load_authors_data(data_dir_val)
        _authors_ok = True
    except Exception as _e:
        st.warning(f"Authors data unavailable: {_e}")
        authors_df = pd.DataFrame(columns=["author_id", "author_name"])
        _authors_ok = False

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Top Authors")
        if _authors_ok and "author_id" in seed.columns and not seed["author_id"].isna().all():
            top_auth = (seed.explode("author_id")
                        .merge(authors_df, on="author_id", how="left")
                        .groupby("author_name").size()
                        .reset_index(name="paper_count")
                        .sort_values("paper_count", ascending=False).head(20))
        else:
            top_auth = (seed["author"].value_counts()
                        .reset_index().rename(columns={"author":"author_name","count":"paper_count"})
                        .head(20))
        top_auth = top_auth[top_auth["author_name"].str.strip() != ""]
        st.plotly_chart(
            px.bar(top_auth, x="paper_count", y="author_name", orientation="h",
                   title="Top 20 Authors")
            .update_layout(yaxis=dict(autorange="reversed"),
                           xaxis_title="Seed Papers", yaxis_title=""),
            use_container_width=True)

    with col_b:
        st.subheader("Top Journals")
        top_jnl = (seed.groupby("journal").size()
                   .reset_index(name="count").sort_values("count", ascending=False).head(20))
        top_jnl = top_jnl[top_jnl["journal"].str.strip() != ""]
        st.plotly_chart(
            px.bar(top_jnl, x="count", y="journal", orientation="h",
                   title="Top 20 Journals")
            .update_layout(yaxis=dict(autorange="reversed"),
                           xaxis_title="Seed Papers", yaxis_title=""),
            use_container_width=True)

    st.markdown("---")
    col_c, col_d = st.columns(2)

    with col_c:
        st.subheader("CitationHub Field × Intent Distribution Heatmap")
        fi = (seed[["seed_paper_id","field"]]
              .merge(events[["seed_paper_id","primary_intent"]], on="seed_paper_id", how="inner")
              .groupby(["field","primary_intent"]).size().reset_index(name="count"))
        if not fi.empty:
            pivot = fi.pivot(index="field", columns="primary_intent", values="count").fillna(0)
            st.plotly_chart(
                px.imshow(pivot, color_continuous_scale="Blues",
                          title="CitationHub Field × Intent Distribution Heatmap",
                          aspect="auto")
                .update_layout(xaxis_title="Intent", yaxis_title="Field"),
                use_container_width=True)

    with col_d:
        st.subheader("Influential Citations (selected paper)")
        if "is_influential" in seed_events.columns:
            inf = seed_events["is_influential"].value_counts().reset_index()
            inf.columns = ["is_influential","count"]
            inf["label"] = inf["is_influential"].map({True:"Influential", False:"Non-influential"})
            st.plotly_chart(
                px.pie(inf, names="label", values="count",
                       title="Influential vs Non-influential"),
                use_container_width=True)

    st.markdown("---")
    st.subheader("CitationHub Intent Evolution over Years")
    st.caption("How citation intents have changed across all papers over time")
    intent_trend_raw = (
        events.dropna(subset=["citing_year"])
        .assign(year=lambda df: df["citing_year"].astype(int))
        .query("year >= 2000")
        .groupby(["year", "primary_intent"]).size()
        .reset_index(name="count")
    )
    if not intent_trend_raw.empty:
        st.plotly_chart(
            px.area(
                intent_trend_raw, x="year", y="count", color="primary_intent",
                color_discrete_map=INTENT_COLORS,
                labels={"primary_intent": "Intent", "count": "Citations", "year": "Year"},
            ).update_layout(
                legend_title="Intent",
                xaxis_title="Year", yaxis_title="# Citations",
                hovermode="x unified",
            ),
            use_container_width=True,
        )

    st.markdown("---")
    col_v1, col_v2 = st.columns(2)

    with col_v1:
        st.subheader("Top Citing Venues")
        st.caption("Journals/conferences that cite seed papers most")
        venue_cnt = (
            events[events["citing_venue"].str.strip() != ""]
            .groupby("citing_venue").size()
            .reset_index(name="count")
            .sort_values("count", ascending=False).head(20)
        )
        if not venue_cnt.empty:
            st.plotly_chart(
                px.bar(venue_cnt, x="count", y="citing_venue", orientation="h",
                       labels={"count": "Citations", "citing_venue": ""})
                .update_layout(yaxis=dict(autorange="reversed"),
                               xaxis_title="Citations", yaxis_title="", height=520),
                use_container_width=True,
            )

    with col_v2:
        st.subheader("CitationHub Field × Intent Distribution")
        st.caption("How each field uses citations differently (all fields)")
        fi_pct = (
            seed[["seed_paper_id", "field"]]
            .merge(events[["seed_paper_id", "primary_intent"]], on="seed_paper_id", how="inner")
            .groupby(["field", "primary_intent"]).size().reset_index(name="count")
        )
        if not fi_pct.empty:
            totals = fi_pct.groupby("field")["count"].transform("sum")
            fi_pct["pct"] = (fi_pct["count"] / totals * 100).round(1)
            n_fields = fi_pct["field"].nunique()
            chart_height = max(520, n_fields * 28)
            st.plotly_chart(
                px.bar(fi_pct, x="pct", y="field", color="primary_intent",
                       orientation="h", color_discrete_map=INTENT_COLORS,
                       labels={"pct": "% of citations", "field": "", "primary_intent": "Intent"})
                .update_layout(
                    barmode="stack",
                    yaxis=dict(autorange="reversed", categoryorder="total ascending"),
                    xaxis_title="% of citations", yaxis_title="",
                    legend_title="Intent", height=chart_height,
                ),
                use_container_width=True,
            )

    st.markdown("---")
    st.subheader("Citation Trend over Time (selected paper)")
    st.caption("How citations to the selected seed paper have changed year by year")
    trend_sel = (seed_events.dropna(subset=["citing_year"])
                 .assign(citing_year=lambda df: df["citing_year"].astype(int))
                 .query("citing_year >= 2000")
                 .groupby("citing_year").size().reset_index(name="count"))
    if not trend_sel.empty:
        st.plotly_chart(
            px.line(trend_sel, x="citing_year", y="count", markers=True,
                    labels={"citing_year": "Year", "count": "Citations"})
            .update_layout(xaxis_title="Year", yaxis_title="Citations",
                           hovermode="x unified"),
            use_container_width=True)
    else:
        st.info("No citation trend data for the selected paper.")

    st.markdown("---")
    st.subheader("Export Data")
    col_e1, col_e2, col_e3 = st.columns(3)

    with col_e1:
        csv_seed = seed_filtered[
            ["title", "doi", "journal", "author", "country", "field", "citedby_count"]
        ].to_csv(index=False).encode("utf-8")
        csv_download_link(csv_seed, "seed_papers.csv", "⬇ Seed Papers (CSV)")

    with col_e2:
        _cite_cols = [c for c in
            ["citing_title", "citing_doi", "citing_year", "citing_venue",
             "primary_intent", "context_count", "is_influential"]
            if c in seed_events.columns]
        cite_export = (seed_events[_cite_cols]
            .rename(columns={
                "citing_title": "title", "citing_doi": "doi",
                "citing_year": "year", "citing_venue": "venue",
                "primary_intent": "intent", "context_count": "contexts",
                "is_influential": "influential",
            }).to_csv(index=False).encode("utf-8"))
        csv_download_link(cite_export, "citation_events.csv", "⬇ Citation Events (CSV)")

    with col_e3:
        intent_csv = intent_summary.to_csv(index=False).encode("utf-8")
        csv_download_link(intent_csv, "intent_summary.csv", "⬇ Intent Summary (CSV)")
