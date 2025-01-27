from sklearn.manifold import TSNE
from sentence_transformers import SentenceTransformer
import plotly.graph_objects as go

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_embeddings(texts):
    """
    Generate a 2D visualization of text embeddings with refined styling.
    """
    try:
        # Generate embeddings
        embeddings = model.encode(texts)
        perplexity = min(30, len(texts) - 1)
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        reduced_embeddings = tsne.fit_transform(embeddings)
        
        # Calculate appropriate ranges with padding
        x_min, x_max = reduced_embeddings[:, 0].min(), reduced_embeddings[:, 0].max()
        y_min, y_max = reduced_embeddings[:, 1].min(), reduced_embeddings[:, 1].max()
        x_padding = (x_max - x_min) * 0.2
        y_padding = (y_max - y_min) * 0.2
        
        # Create figure
        fig = go.Figure()
        
        # Add scatter points
        fig.add_trace(go.Scatter(
            x=reduced_embeddings[:, 0],
            y=reduced_embeddings[:, 1],
            mode='markers+text',
            text=texts,
            textposition="top center",
            marker=dict(
                size=12,
                color='#C48F8F',
                line=dict(width=2, color='#A67676')
            ),
            textfont=dict(
                family='Inter',
                size=14,
                color='#333333'
            )
        ))

        # Update layout with increased margins
        fig.update_layout(
            title=dict(
                text="Text Embedding Visualization",
                font=dict(family="Inter", size=20, color="#333333"),
                x=0.5,
                y=0.95
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            width=None,
            height=700,  # Increased height
            margin=dict(l=40, r=40, t=60, b=200),  # Increased bottom margin
            xaxis=dict(
                title="Dimension 1",
                gridcolor='#E0E0E0',
                showgrid=True,
                zeroline=False,
                titlefont=dict(family="Inter", size=14, color="#333333"),
                tickfont=dict(family="Inter", size=12, color="#333333"),
                range=[x_min - x_padding, x_max + x_padding],
            ),
            yaxis=dict(
                title="Dimension 2",
                gridcolor='#E0E0E0',
                showgrid=True,
                zeroline=False,
                titlefont=dict(family="Inter", size=14, color="#333333"),
                tickfont=dict(family="Inter", size=12, color="#333333"),
                range=[y_min - y_padding, y_max + y_padding],
            ),
            showlegend=False
        )

        # Add description text at the bottom with more space
        fig.add_annotation(
            text="<b>Understanding the Visualization:</b><br>" +
                 "• Points that are closer together have similar meanings<br>" +
                 "• Distance between points represents semantic difference<br>" +
                 "• Clusters indicate groups of related concepts",
            align="left",
            showarrow=False,
            xref="paper",
            yref="paper",
            x=0,
            y=-0.35,  # Moved lower
            bordercolor="#C48F8F",
            borderwidth=2,
            borderpad=10,
            bgcolor="rgba(255, 255, 255, 0.95)",
            font=dict(family="Inter", size=12, color="#333333")
        )

        return fig
    except Exception as e:
        fig = go.Figure()
        fig.update_layout(
            title=dict(text="Error Generating Embeddings", font=dict(family="Inter", size=20, color="#D64545")),
            height=700,
            annotations=[dict(
                text=f"Error: {str(e)}",
                xref="paper", yref="paper",
                x=-0.1, y=0.5,
                showarrow=False,
                font=dict(family="Inter", size=14, color="#D64545"),
                bgcolor="rgba(255, 255, 255, 0.95)",
                bordercolor="#D64545",
                borderwidth=2,
                borderpad=10
            )]
        )
        return fig