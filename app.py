import gradio as gr
from src.embeddings.embeddings import generate_embeddings
from src.attention.attention import generate_attention_heatmap

def create_dashboard():
    custom_css = """
    /* Force a consistent light theme, ignoring system dark mode: */
    html, body, .gradio-container {
        background-color: #FDF5F5 !important;
        color: #333333 !important;
        forced-color-adjust: none !important; /* Prevent color inversion in dark mode */
        -webkit-print-color-adjust: exact !important;
        -webkit-font-smoothing: antialiased;
        font-family: 'Inter', sans-serif;
    }

    /* Make sure all child elements also use dark text */
    .gradio-container * {
        color: #333333 !important;
    }

    .container {
        max-width: 1000px;
        margin: 0 auto;
        padding: 20px;
    }

    .plot-container {
        background: white;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        padding: 20px;
        margin: 10px 0;
    }

    .textbox {
    border: 2px solid #C48F8F !important;
    border-radius: 8px;
    background: white;
    font-size: 16px;
    margin: 10px 0 !important;
    /* We'll still default the outer box to white with dark text */
    color: #333333 !important;
    }

    /* Force the actual <textarea> / <input> inside the gr.Textbox to have 
    a light background with dark text, regardless of OS dark mode. */
    .textbox textarea, .textbox input {
        background-color: #FDF5F5 !important;  /* or just 'white' if you prefer */
        color: #333333 !important;
        forced-color-adjust: none !important;  /* prevents system color inversion */
        -webkit-text-fill-color: #333333 !important; /* sometimes needed on Safari */
    }

    /* Optionally style placeholder text too, so it's visible/legible */
    .textbox textarea::placeholder,
    .textbox input::placeholder {
        color: #999999 !important;
    }


    button.primary {
        background: #C48F8F !important;
        border: none !important;
        box-shadow: 0 2px 6px rgba(196, 143, 143, 0.4) !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        padding: 10px 20px !important;
        margin: 10px 0 !important;
        color: white !important;
        forced-color-adjust: none !important;
    }
    """

    with gr.Blocks(css=custom_css) as demo:
        with gr.Column(elem_classes="container"):
            gr.Markdown("# ✨ Interactive Model Explainability Dashboard")
            
            with gr.Tabs() as tabs:
                with gr.Tab("📊 Text Embeddings"):
                    with gr.Column(elem_classes="plot-container"):
                        gr.Markdown(
                            "Embeddings encode text into numerical vectors, allowing models to understand similarity. "
                            "Below, you can visualize your text inputs in 2D space using t-SNE.",
                            elem_classes="markdown-text"
                        )
                        embedding_plot = gr.Plot(label="Text Embedding Visualization")
                        with gr.Column(elem_classes="input-container"):
                            text_input = gr.Textbox(
                                lines=2,
                                placeholder="Enter texts separated by commas (e.g. blazer, elf, mage)",
                                label="Enter Texts",
                                elem_classes="textbox"
                            )
                            gr.Button(
                                "✨ Generate Embeddings", 
                                variant="primary",
                                size="sm"
                            ).click(
                                lambda texts: generate_embeddings([t.strip() for t in texts.split(",") if t.strip()]),
                                inputs=text_input,
                                outputs=embedding_plot
                            )

                with gr.Tab("🔥 Attention Heatmap"):
                    with gr.Column(elem_classes="plot-container"):
                        gr.Markdown(
                            "Explore how BERT distributes attention across tokens. "
                            "We'll visualize the final-layer attention (averaged over all heads) as a heatmap. "
                            "You can remove special tokens and unify subwords for readability.",
                            elem_classes="markdown-text"
                        )
                        attention_plot = gr.Plot(label="Attention Heatmap")
                        with gr.Column(elem_classes="input-container"):
                            text_input = gr.Textbox(
                                lines=2,
                                placeholder="Enter a sentence (e.g. I love toast with jam)",
                                label="Enter a Sentence",
                                elem_classes="textbox"
                            )
                            gr.Button(
                                "✨ Generate Heatmap", 
                                variant="primary",
                                size="sm"
                            ).click(
                                lambda text: generate_attention_heatmap(text.strip()) if text.strip() else None,
                                inputs=text_input,
                                outputs=attention_plot
                            )

    return demo

if __name__ == "__main__":
    app = create_dashboard()
    app.launch()
