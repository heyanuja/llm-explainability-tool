from sklearn.manifold import TSNE
from sentence_transformers import SentenceTransformer
import plotly.graph_objects as go


model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_embeddings(texts):
    """
    Generate a 2D t-SNE scatter plot from MiniLM embeddings.
    """
    try:
        # Input validation
        if not texts:
            raise ValueError("No texts provided for embedding generation")
            
        # Encode texts to embeddings
        embeddings = model.encode(texts)
        
        # t-SNE requires at least 2 samples, handle edge case
        if len(texts) < 2:
            # If only one text, create a simple 2D plot without t-SNE
            reduced_embeddings = [[0, 0]]
        else:
            # Calculate appropriate perplexity (must be less than n_samples - 1)
            perplexity = min(30, len(texts) - 1)
            # Reduce dimensionality
            tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
            reduced_embeddings = tsne.fit_transform(embeddings)
        
        x_min, x_max = reduced_embeddings[:, 0].min(), reduced_embeddings[:, 0].max()
        y_min, y_max = reduced_embeddings[:, 1].min(), reduced_embeddings[:, 1].max()
        x_padding = (x_max - x_min) * 0.2
        y_padding = (y_max - y_min) * 0.2

        fig = go.Figure()
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

        fig.update_layout(
            template=None,
            title=dict(
                text="Text Embedding Visualization",
                font=dict(family="Inter", size=20, color="#333333"),
                x=0.5,
                y=0.95
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            width=None,
            height=700,
            margin=dict(l=40, r=40, t=60, b=200),
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

        fig.add_annotation(
            text="<b>Understanding the Visualization:</b><br>"
                 "• Points closer together often have similar meanings<br>"
                 "• Distance roughly represents semantic difference<br>"
                 "• Clusters indicate groups of related concepts",
            align="left",
            showarrow=False,
            xref="paper",
            yref="paper",
            x=0,
            y=-0.35,
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
            template=None,
            title=dict(
                text="Error Generating Embeddings",
                font=dict(family="Inter", size=20, color="#D64545"
            )),
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
