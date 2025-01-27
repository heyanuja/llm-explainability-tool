import numpy as np
import plotly.graph_objects as go

def generate_attention_heatmap(text):
    """
    Generate a heatmap to visualize attention weights for a sentence of your choosing!
    """
    try:
        tokens = text.split()
        n_tokens = len(tokens)
        Q, K = embedding(text)
        attention_unscaled = Q @ K.T
        attention_weights = torch.softmax(attention_unscaled / sqrt(384), dim=-1)
        attention_weights = attention_weights.numpy()
        fig = go.Figure()

        # Add heatmap
        fig.add_trace(go.Heatmap(
            z=attention_weights,
            x=tokens,
            y=tokens,
            colorscale=[
                [0, "#FDF5F5"],
                [0.5, "#E8D0D0"],
                [1, "#C48F8F"]
            ],
            colorbar=dict(
                title="Attention Strength",
                titlefont=dict(family="Inter", size=14, color="#333333"),
                tickfont=dict(family="Inter", size=12, color="#333333"),
                thickness=15,
                y=0.6,
                len=0.8
            ),
            hoverongaps=False,
            hoverinfo="z+x+y"
        ))

        # Update layout with increased margins
        fig.update_layout(
            title=dict(
                text="Attention Heatmap",
                font=dict(family="Inter", size=20, color="#333333"),
                x=0.5,
                y=0.95
            ),
            width=None,
            height=800,  # Increased height
            margin=dict(l=40, r=40, t=60, b=300),  # Increased bottom margin
            xaxis=dict(
                title="   Key Tokens",
                tickangle=45,
                tickfont=dict(size=12, color="#333333", family="Inter"),
                title_standoff=20,
                title_font=dict(size=14, color="#333333", family="Inter")
            ),
            yaxis=dict(
                title="Query Tokens",
                tickfont=dict(size=12, color="#333333", family="Inter"),
                title_font=dict(size=14, color="#333333", family="Inter")
            ),
            plot_bgcolor="white",
            paper_bgcolor="white"
        )

        # Add description text at the bottom with more space
        fig.add_annotation(
            text="<b>Understanding Attention:</b><br>" +
                 "• Darker colors show stronger connections between words<br>" +
                 "• Each row shows how a word relates to other words<br>" +
                 "• The pattern reveals the model's focus across the sentence",
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
            title=dict(text="Error Generating Heatmap", font=dict(family="Inter", size=20, color="#D64545")),
            height=700,
            annotations=[dict(
                text=f"Error: {str(e)}",
                xref="paper", yref="paper",
                x=-0.5, y=-0.5,
                showarrow=False,
                font=dict(family="Inter", size=14, color="#D64545"),
                bgcolor="rgba(255, 255, 255, 0.95)",
                bordercolor="#D64545",
                borderwidth=2,
                borderpad=10
            )]
        )
        return fig