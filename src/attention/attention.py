import torch
import plotly.graph_objects as go
from transformers import AutoTokenizer, AutoModel

# Load BERT once at import, with output_attentions=True
_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
_model = AutoModel.from_pretrained("bert-base-uncased", output_attentions=True)

def _clean_bert_tokens(tokens):
    """
    Remove [CLS], [SEP], and strip the leading '##' from subwords to make
    the displayed tokens more user-friendly.
    """
    cleaned = []
    for t in tokens:
        if t in ("[CLS]", "[SEP]"):
            # skip special tokens
            continue
        if t.startswith("##"):
            t = t[2:]  # remove the '##'
        cleaned.append(t)
    return cleaned

def _get_bert_attention(text):
    """
    1. Tokenize input text
    2. Pass through BERT (with output_attentions=True)
    3. Return the final layer's attention averaged across all heads
       plus the BERT tokens themselves.
    """
    inputs = _tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = _model(**inputs)
    # outputs.attentions is a tuple of shape [num_layers].
    # Each entry is (batch_size=1, num_heads, seq_len, seq_len).
    final_layer = outputs.attentions[-1]  # final layer
    # Average over heads => shape: (batch_size=1, seq_len, seq_len)
    attn_avg = final_layer.mean(dim=1)
    # Drop batch dim => shape: (seq_len, seq_len)
    attn_matrix = attn_avg[0]

    # Convert ids back to raw tokens
    input_ids = inputs["input_ids"][0].tolist()
    tokens = _tokenizer.convert_ids_to_tokens(input_ids, skip_special_tokens=False)

    return tokens, attn_matrix

def generate_attention_heatmap(text):
    """
    Generate a Plotly heatmap showing the final-layer attention from BERT.
    We remove [CLS]/[SEP] and strip '##' from subword tokens for readability.
    """
    try:
        # Get raw tokens + attention
        tokens, attn_torch = _get_bert_attention(text)

        # Convert to NumPy before slicing
        attn_weights = attn_torch.cpu().numpy()

        # We'll keep track of which indices are *not* [CLS]/[SEP] so we can filter both tokens & matrix
        # Or simply "clean" the tokens for display. We do it in two steps for clarity:
        original_tokens = tokens
        keep_mask = [i for i, tok in enumerate(original_tokens)
                     if tok not in ("[CLS]", "[SEP]")]

        # Slice out [CLS]/[SEP] from attention matrix
        filtered_matrix = attn_weights[keep_mask, :][:, keep_mask]

        # "Clean" subword tokens for final display
        display_tokens = _clean_bert_tokens(original_tokens)

        # Convert final matrix to 2D array
        # (but note that display_tokens no longer includes [CLS]/[SEP],
        #  so it's consistent with filtered_matrix shape)
        attn_weights = filtered_matrix

        # Build the figure
        fig = go.Figure()

        fig.add_trace(go.Heatmap(
            z=attn_weights,
            x=display_tokens,
            y=display_tokens,
            # Adjust the color scale or keep your pink scale:
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

        # Reverse the y-axis so the first token is at the top
        fig.update_yaxes(autorange="reversed")

        # Force a light background so it doesn't invert in Dark Mode
        fig.update_layout(
            template=None,
            title=dict(
                text="BERT Attention Heatmap",
                font=dict(family="Inter", size=20, color="#333333"),
                x=0.5,
                y=0.95
            ),
            width=None,
            height=800,
            margin=dict(l=40, r=40, t=60, b=300),
            xaxis=dict(
                title="Key Tokens",
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

        # Add descriptive annotation
        fig.add_annotation(
            text="<b>Understanding BERT Attention:</b><br>"
                 "• Darker colors show stronger connections between tokens<br>"
                 "• Each row shows how a token 'attends' to every other token<br>"
                 "• These weights are from the final layer, averaged over heads<br>"
                 "• [CLS] / [SEP] removed for clarity, subwords merged",
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
        # Graceful fallback if something fails
        fig = go.Figure()
        fig.update_layout(
            template=None,
            title=dict(
                text="Error Generating Heatmap",
                font=dict(family="Inter", size=20, color="#D64545")
            ),
            height=700,
            annotations=[dict(
                text=f"Error: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(family="Inter", size=14, color="#D64545"),
                bgcolor="rgba(255, 255, 255, 0.95)",
                bordercolor="#D64545",
                borderwidth=2,
                borderpad=10
            )]
        )
        return fig
