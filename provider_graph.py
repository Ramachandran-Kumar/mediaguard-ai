"""
MediGuard AI — Provider Risk Network Graph
File: provider_graph.py

Builds a NetworkX graph from claims_flagged.csv where:
  - Nodes  = providers (sized by claim count, colored by dominant fraud type)
  - Edges  = shared fraud label patterns OR shared CPT codes between providers
             (thickness proportional to combined edge weight)

Output:
  output/provider_graph.png  — enterprise-style visualization
  Console                    — text summary of nodes, edges, clusters
"""

import os
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from itertools import combinations
from collections import Counter

# ── CONFIG ────────────────────────────────────────────────────────────────────

INPUT_FILE  = "output/claims_flagged.csv"
OUTPUT_FILE = "assets/provider_graph.png"

FRAUD_COLORS = {
    "UPCODING":             "#4C72B0",   # blue
    "UNBUNDLING":           "#C44E52",   # red
    "ICD_CPT_MISMATCH":     "#DD8452",   # orange
    "MEDICALLY_UNNECESSARY":"#9467BD",   # purple
    "CLEAN":                "#55A868",   # green
    "MIXED":                "#8C8C8C",   # gray — fallback
}


# ── DATA LOADING ──────────────────────────────────────────────────────────────

def load_claims(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    # Normalise CPT codes to zero-padded strings (stored as float in CSV)
    df["cpt_str"] = df["cpt_code"].apply(
        lambda x: str(int(x)).zfill(5) if pd.notna(x) else ""
    )
    return df


# ── NODE BUILDING ─────────────────────────────────────────────────────────────

def build_nodes(df: pd.DataFrame) -> dict:
    """
    Returns dict: provider_npi -> attribute dict.
    dominant_fraud_type = most common non-CLEAN fraud label;
    falls back to CLEAN if all claims are clean.
    """
    nodes = {}
    for npi, grp in df.groupby("provider_npi"):
        fraud_counts = Counter(
            label for label in grp["fraud_label"] if label != "CLEAN"
        )
        dominant = fraud_counts.most_common(1)[0][0] if fraud_counts else "CLEAN"

        nodes[npi] = {
            "npi":               npi,
            "name":              grp["provider_name"].iloc[0],
            "specialty":         grp["provider_specialty"].iloc[0],
            "state":             grp["provider_state"].iloc[0],
            "claim_count":       len(grp),
            "dominant_fraud":    dominant,
            "avg_risk_score":    round(grp["composite_risk_score"].mean(), 1),
            "fraud_labels":      set(grp["fraud_label"].unique()),
            "cpt_codes":         set(grp["cpt_str"].unique()),
        }
    return nodes


# ── EDGE BUILDING ─────────────────────────────────────────────────────────────

def build_edges(nodes: dict) -> list[dict]:
    """
    For every pair of providers, compute:
      fraud_weight = number of shared fraud labels (excluding CLEAN)
      cpt_weight   = number of shared CPT codes
      total_weight = fraud_weight + cpt_weight

    Only creates an edge if total_weight > 0.
    """
    edges = []
    npis = list(nodes.keys())

    for npi_a, npi_b in combinations(npis, 2):
        a, b = nodes[npi_a], nodes[npi_b]

        shared_fraud = {
            label for label in (a["fraud_labels"] & b["fraud_labels"])
            if label != "CLEAN"
        }
        shared_cpts  = a["cpt_codes"] & b["cpt_codes"] - {""}

        fw = len(shared_fraud)
        cw = len(shared_cpts)
        tw = fw + cw

        if tw > 0:
            edges.append({
                "src":          npi_a,
                "dst":          npi_b,
                "fraud_weight": fw,
                "cpt_weight":   cw,
                "total_weight": tw,
                "shared_fraud": shared_fraud,
                "shared_cpts":  shared_cpts,
            })

    return edges


# ── GRAPH CONSTRUCTION ────────────────────────────────────────────────────────

def build_graph(nodes: dict, edges: list[dict]) -> nx.Graph:
    G = nx.Graph()

    for npi, attrs in nodes.items():
        G.add_node(npi, **attrs)

    for e in edges:
        G.add_edge(
            e["src"], e["dst"],
            weight=e["total_weight"],
            fraud_weight=e["fraud_weight"],
            cpt_weight=e["cpt_weight"],
            shared_fraud=e["shared_fraud"],
            shared_cpts=e["shared_cpts"],
        )

    return G


# ── VISUALIZATION ─────────────────────────────────────────────────────────────

def make_figure(G: nx.Graph) -> plt.Figure:
    """
    Build and return the matplotlib Figure for the provider graph.
    Called by draw_graph() (save to file) and by the Streamlit dashboard
    (display inline via st.pyplot).
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("white")
    _render_graph_onto(G, fig, ax)
    plt.tight_layout()
    return fig


def _render_graph_onto(G: nx.Graph, fig: plt.Figure, ax):
    """Core drawing logic — shared by make_figure() and draw_graph()."""
    ax.set_facecolor("#F8F9FA")

    # Layout — Kamada-Kawai keeps high-weight edges short (clusters emerge)
    pos = nx.kamada_kawai_layout(G, weight="weight")

    # ── Node visuals ──────────────────────────────────────────────────────────
    node_colors = [
        FRAUD_COLORS.get(G.nodes[n]["dominant_fraud"], FRAUD_COLORS["MIXED"])
        for n in G.nodes
    ]
    # Scale node size: min 800, max 3200, proportional to claim_count
    claim_counts  = [G.nodes[n]["claim_count"] for n in G.nodes]
    min_c, max_c  = min(claim_counts), max(claim_counts)
    size_range    = (3200 - 800) / max(max_c - min_c, 1)
    node_sizes    = [800 + (G.nodes[n]["claim_count"] - min_c) * size_range
                     for n in G.nodes]

    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.92,
        linewidths=1.8,
        edgecolors="white",
    )

    # ── Edge visuals ──────────────────────────────────────────────────────────
    if G.edges:
        weights     = [G[u][v]["weight"] for u, v in G.edges]
        max_w       = max(weights)
        edge_widths = [1.0 + (w / max_w) * 5.5 for w in weights]
        edge_alphas = [0.35 + (w / max_w) * 0.50 for w in weights]

        # Draw edges one-by-one so each can have individual alpha
        for (u, v), width, alpha in zip(G.edges, edge_widths, edge_alphas):
            nx.draw_networkx_edges(
                G, pos, ax=ax,
                edgelist=[(u, v)],
                width=width,
                alpha=alpha,
                edge_color="#555555",
                style="solid",
            )

    # ── Node labels ───────────────────────────────────────────────────────────
    labels = {
        n: f"{G.nodes[n]['name'].replace('Dr. ', '')}\n"
           f"{G.nodes[n]['dominant_fraud']}\n"
           f"Risk: {G.nodes[n]['avg_risk_score']}"
        for n in G.nodes
    }
    nx.draw_networkx_labels(
        G, pos, labels, ax=ax,
        font_size=7.5,
        font_weight="bold",
        font_color="#1A1A1A",
    )

    # ── Edge weight labels (on strong edges only) ─────────────────────────────
    strong_edges = {
        (u, v): f"w={G[u][v]['weight']}"
        for u, v in G.edges
        if G[u][v]["weight"] >= 2
    }
    nx.draw_networkx_edge_labels(
        G, pos, strong_edges, ax=ax,
        font_size=6.5,
        font_color="#333333",
        bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.7, ec="none"),
    )

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_patches = [
        mpatches.Patch(color=color, label=label)
        for label, color in FRAUD_COLORS.items()
        if label != "MIXED"
    ]
    ax.legend(
        handles=legend_patches,
        title="Dominant Fraud Type",
        title_fontsize=9,
        fontsize=8,
        loc="lower left",
        framealpha=0.92,
        edgecolor="#CCCCCC",
        fancybox=True,
    )

    # ── Titles & chrome ───────────────────────────────────────────────────────
    ax.set_title(
        "MediGuard AI — Provider Risk Network",
        fontsize=16, fontweight="bold", color="#1A1A1A", pad=18,
    )
    ax.text(
        0.5, -0.03,
        f"Nodes: {G.number_of_nodes()} providers  |  "
        f"Edges: {G.number_of_edges()} connections  |  "
        "Edge weight = shared fraud patterns + shared CPT codes",
        transform=ax.transAxes,
        ha="center", fontsize=8, color="#666666",
    )

    # Thin border around axes
    for spine in ax.spines.values():
        spine.set_edgecolor("#DDDDDD")
        spine.set_linewidth(0.8)

    ax.axis("off")
    plt.tight_layout(pad=1.5)


def draw_graph(G: nx.Graph, output_path: str):
    """Build the figure, save to disk, and close."""
    fig = make_figure(G)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n  Graph saved → {output_path}")


# ── TEXT SUMMARY ──────────────────────────────────────────────────────────────

def print_summary(G: nx.Graph, edges: list[dict]):
    print("\n" + "═" * 65)
    print("  MEDIAGUARD AI — PROVIDER RISK NETWORK SUMMARY")
    print("═" * 65)
    print(f"  Total nodes (providers)  : {G.number_of_nodes()}")
    print(f"  Total edges (connections): {G.number_of_edges()}")

    if not edges:
        print("  No edges found.")
        return

    # ── Nodes ─────────────────────────────────────────────────────────────────
    print("\n  PROVIDER NODES:")
    print(f"  {'Provider':<22} {'Fraud Type':<22} {'Claims':>6}  {'Avg Risk':>8}  State")
    print(f"  {'─'*22} {'─'*22} {'─'*6}  {'─'*8}  {'─'*5}")
    for n in sorted(G.nodes, key=lambda n: G.nodes[n]["avg_risk_score"], reverse=True):
        d = G.nodes[n]
        print(f"  {d['name']:<22} {d['dominant_fraud']:<22} {d['claim_count']:>6}  "
              f"{d['avg_risk_score']:>7.1f}  {d['state']}")

    # ── Strongest edges ───────────────────────────────────────────────────────
    sorted_edges = sorted(edges, key=lambda e: e["total_weight"], reverse=True)
    print(f"\n  STRONGEST CONNECTIONS (by total weight):")
    print(f"  {'Provider A':<22} {'Provider B':<22} {'Total':>5}  {'Fraud':>5}  {'CPT':>5}  Shared")
    print(f"  {'─'*22} {'─'*22} {'─'*5}  {'─'*5}  {'─'*5}  {'─'*30}")
    for e in sorted_edges[:10]:
        name_a = G.nodes[e["src"]]["name"]
        name_b = G.nodes[e["dst"]]["name"]
        shared_detail = (
            (", ".join(sorted(e["shared_fraud"])) if e["shared_fraud"] else "") +
            (" | CPT: " + ", ".join(sorted(e["shared_cpts"])) if e["shared_cpts"] else "")
        ).strip(" |")
        print(f"  {name_a:<22} {name_b:<22} {e['total_weight']:>5}  "
              f"{e['fraud_weight']:>5}  {e['cpt_weight']:>5}  {shared_detail}")

    # ── Cluster summary ───────────────────────────────────────────────────────
    components = list(nx.connected_components(G))
    print(f"\n  CLUSTER SUMMARY ({len(components)} connected component(s)):")
    for i, comp in enumerate(sorted(components, key=len, reverse=True), 1):
        members = sorted(comp, key=lambda n: G.nodes[n]["avg_risk_score"], reverse=True)
        fraud_types = sorted({G.nodes[n]["dominant_fraud"] for n in members})
        print(f"\n  Cluster {i} — {len(members)} provider(s), "
              f"fraud types: {', '.join(fraud_types)}")
        for n in members:
            d = G.nodes[n]
            degree = G.degree(n)
            print(f"    • {d['name']:<22}  {d['dominant_fraud']:<22}  "
                  f"risk={d['avg_risk_score']}  degree={degree}")

    # ── Degree centrality (most connected providers) ──────────────────────────
    if G.number_of_edges() > 0:
        centrality = nx.degree_centrality(G)
        top = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"\n  MOST CONNECTED PROVIDERS (degree centrality):")
        for npi, score in top:
            d = G.nodes[npi]
            print(f"    {d['name']:<22}  centrality={score:.2f}  "
                  f"connections={G.degree(npi)}")

    print("\n" + "═" * 65)


# ── CLUSTER ALERTS (shared by dashboard and CLI) ─────────────────────────────

def get_cluster_alerts(G: nx.Graph) -> list[dict]:
    """
    Returns a list of alert dicts for every connected cluster with >1 provider.
    Used by the Streamlit dashboard and by print_scheme_narratives().
    """
    components  = sorted(nx.connected_components(G), key=len, reverse=True)
    alerts      = []

    for cluster in components:
        if len(cluster) <= 1:
            continue

        members        = list(cluster)
        states         = sorted({G.nodes[n]["state"] for n in members})
        fraud_counts   = Counter(G.nodes[n]["dominant_fraud"] for n in members)
        dominant_fraud = fraud_counts.most_common(1)[0][0]
        total_claims   = sum(G.nodes[n]["claim_count"] for n in members)
        avg_risk       = sum(G.nodes[n]["avg_risk_score"] for n in members) / len(members)
        provider_names = sorted(G.nodes[n]["name"] for n in members)

        shared_cpts: set = set()
        shared_fraud_labels: set = set()
        for u, v in G.edges(members):
            if v in cluster:
                shared_cpts         |= G[u][v].get("shared_cpts", set())
                shared_fraud_labels |= G[u][v].get("shared_fraud", set())
        shared_cpts.discard("")

        if avg_risk >= 80:
            action = "Joint SIU Investigation — refer to OIG"
        elif avg_risk >= 60:
            action = "Joint SIU Investigation with prepayment audit"
        else:
            action = "Enhanced monitoring — flag for coordinated review"

        alerts.append({
            "provider_count":      len(members),
            "state_count":         len(states),
            "states":              states,
            "dominant_fraud":      dominant_fraud,
            "shared_fraud_labels": shared_fraud_labels,
            "shared_cpts":         shared_cpts,
            "total_claims":        total_claims,
            "avg_risk":            round(avg_risk, 1),
            "provider_names":      provider_names,
            "action":              action,
        })

    return alerts


# ── SCHEME NARRATIVE ──────────────────────────────────────────────────────────

def print_scheme_narratives(G: nx.Graph):
    """
    Print investigator-language scheme alerts for each multi-provider cluster.
    Uses get_cluster_alerts() so logic is shared with the Streamlit dashboard.
    """
    alerts = get_cluster_alerts(G)

    if not alerts:
        print("\n  No multi-provider clusters detected.")
        return

    fraud_notes = {
        "UNBUNDLING":             "Review NCCI edits and EOB patterns for systematic code-splitting across all named providers.",
        "UPCODING":               "Pull E&M documentation and compare complexity justification against diagnosis severity.",
        "ICD_CPT_MISMATCH":       "Request operative and clinical records to verify procedure-diagnosis alignment.",
        "MEDICALLY_UNNECESSARY":  "Obtain clinical records and prior-auth history to assess medical necessity.",
    }

    print("\n" + "█" * 65)
    print("  MEDIAGUARD AI — SCHEME-LEVEL CLUSTER ALERTS")
    print("█" * 65)

    for i, alert in enumerate(alerts, 1):
        cpt_str   = ", ".join(sorted(alert["shared_cpts"])) if alert["shared_cpts"] else "none identified"
        fraud_str = ", ".join(sorted(alert["shared_fraud_labels"])) if alert["shared_fraud_labels"] else alert["dominant_fraud"]
        note      = fraud_notes.get(alert["dominant_fraud"], "Conduct multi-provider claims audit.")

        print(f"""
  ┌─ CLUSTER ALERT {i} {'─' * (47 - len(str(i)))}┐

  Cluster Alert: {alert["provider_count"]} providers across {alert["state_count"]} state(s) show a coordinated
  {alert["dominant_fraud"]} billing pattern.

  Providers involved : {', '.join(alert["provider_names"])}
  States             : {', '.join(alert["states"])}
  Shared fraud types : {fraud_str}
  Shared CPT codes   : {cpt_str}
  Aggregate claims   : {alert["total_claims"]}
  Average risk score : {alert["avg_risk"]:.1f}/100

  Recommended action : {alert["action"]}
  Investigation note : {note}

  └{'─' * 63}┘""")

    print("\n" + "█" * 65)


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  MediGuard AI — Provider Risk Network Graph              ║")
    print("╚══════════════════════════════════════════════════════════╝")

    print(f"\n  Loading claims from {INPUT_FILE}...")
    df    = load_claims(INPUT_FILE)
    print(f"  {len(df)} claims, {df['provider_npi'].nunique()} providers")

    print("  Building nodes...")
    nodes = build_nodes(df)

    print("  Building edges...")
    edges = build_edges(nodes)

    print(f"  Graph: {len(nodes)} nodes, {len(edges)} edges")

    G = build_graph(nodes, edges)

    print("  Rendering graph...")
    draw_graph(G, OUTPUT_FILE)

    print_summary(G, edges)
    print_scheme_narratives(G)
