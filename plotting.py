import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as PathEffects
import os
import numpy as np

class NetworkPlotter:
    def __init__(self, omics_data_dict):
        """
        Initialize the network plotter.
        :param omics_data_dict: Dictionary of raw omics data
                               {'cytokines': df, ...}. Used to infer which
                               omics layer each feature/node belongs to.
        """
        self.node_type_map = {}
        self._build_node_map(omics_data_dict)
        self.G = nx.Graph()
        
        # Default high-impact‐journal color palette
        self.COLOR_PALETTE = {
            'cytokines': '#E64B35',     # Vermilion red
            'proteomics': '#4DBBD5',    # Sky blue
            'transcriptome': '#00A087', # Jade green
            'unknown': '#B09C85'        # Neutral gray-brown
        }

    def _build_node_map(self, data_dict):
        """Create a mapping {feature_name: omics_type} from column names."""
        for omics_name, df in data_dict.items():
            # Assume first two columns are ID and Time; remaining columns are features
            features = df.columns[2:]
            for f in features:
                self.node_type_map[f] = omics_name

    def build_network_from_file(self, edge_file, target_category="Subtle_Coordinated"):
        """
        Build a network from a post-hoc results file.
        :param edge_file: Path to classified_edges.csv
        :param target_category: Keep only edges of this category (default: 'Subtle_Coordinated')
        """
        df = pd.read_csv(edge_file)
        
        # Filter target category (e.g., draw only Subtle Coordinated)
        if target_category:
            df = df[df['Category'] == target_category]
        
        # Add edges
        edges = list(zip(df['feature_g'], df['feature_h']))
        self.G.add_edges_from(edges)
        print(f"Network built: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges")

    def plot_top_hubs(self, output_file, top_k=20, figsize=(10, 10)):
        """
        Plot a Top-K hub network (Nature-style aesthetic).
        """
        if self.G.number_of_nodes() == 0:
            print("⚠️ Network is empty. Cannot plot.")
            return

        # 1. Select top-K hubs within each omics layer
        node_degrees = dict(self.G.degree())
        selected_nodes = set()
        
        # Identify which omics layers are present in the current network
        present_layers = set(self.node_type_map.get(n, 'unknown') for n in self.G.nodes())
        
        for omics in present_layers:
            layer_nodes = [n for n in self.G.nodes() if self.node_type_map.get(n) == omics]
            
            # Sort by degree and take top-K
            top_nodes = sorted(layer_nodes, key=lambda x: node_degrees[x], reverse=True)[:top_k]
            selected_nodes.update(top_nodes)
            
        if not selected_nodes:
            print("⚠️ No nodes left after selection.")
            return

        # Extract subgraph for plotting
        G_visual = self.G.subgraph(selected_nodes).copy()
        print(f"Plotting subnetwork: {G_visual.number_of_nodes()} nodes (Top {top_k}/layer)")

        # 2. Layout and styling
        plt.figure(figsize=figsize, dpi=300)
        pos = nx.kamada_kawai_layout(G_visual, scale=2.0)
        
        node_colors = []
        node_sizes = []
        labels = {}
        
        degrees = [node_degrees[n] for n in G_visual.nodes()]
        min_deg = min(degrees)
        max_deg = max(degrees) if degrees else 1

        for node in G_visual.nodes():
            # Color by omics type
            otype = self.node_type_map.get(node, "unknown")
            node_colors.append(self.COLOR_PALETTE.get(otype, '#B09C85'))
            
            # Size (normalized between 300–1500)
            deg = node_degrees[node]
            norm_size = (deg - min_deg) / (max_deg - min_deg + 1e-5)
            node_sizes.append(300 + norm_size * 1200)
            
            labels[node] = node

        # 3. Draw edges
        nx.draw_networkx_edges(
            G_visual, pos, width=1.0, alpha=0.2, edge_color='#95a5a6'
        )
        
        # Draw nodes
        nx.draw_networkx_nodes(
            G_visual, pos, node_color=node_colors, node_size=node_sizes,
            alpha=0.95, linewidths=2.0, edgecolors='white'
        )
        
        # Labels with halo effect
        text_items = nx.draw_networkx_labels(
            G_visual, pos, labels,
            font_size=9, font_weight='bold', font_family='Arial',
            horizontalalignment='center', verticalalignment='center'
        )
        for t in text_items.values():
            t.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white', alpha=0.7)])

        # 4. Dynamic legend
        used_layers = set(self.node_type_map.get(n) for n in G_visual.nodes())
        legend_handles = [
            mpatches.Patch(color=self.COLOR_PALETTE.get(l, '#B09C85'), label=l.capitalize())
            for l in used_layers if l in self.COLOR_PALETTE
        ]
        
        plt.legend(
            handles=legend_handles, loc='lower right', fontsize=11,
            frameon=True, edgecolor='none', facecolor='white', framealpha=0.9,
            title="Omics Layer", title_fontsize=12, bbox_to_anchor=(1, 0)
        )
        
        plt.title(
            f"Core Regulatory Backbone (Top {top_k} Hubs)",
            fontsize=16, fontweight='bold', y=0.98
        )
        plt.axis('off')
        
        # Save to file
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✨ Saved figure: {output_file}")
        plt.close()