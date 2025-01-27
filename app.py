import gradio as gr
from src.embeddings.embeddings import generate_embeddings
from src.attention.attention import generate_attention_heatmap

def create_dashboard():
    custom_css = """
    .gradio-container {
        background-color: #FDF5F5;
        font-family: 'Inter', sans-serif;
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
    }
    """

    with gr.Blocks(css=custom_css) as demo:
        with gr.Column(elem_classes="container"):
            gr.Markdown("# ✨ Interactive Model Explainability Dashboard")
            
            with gr.Tabs() as tabs:
                with gr.Tab("📊 Text Embeddings"):
                    with gr.Column(elem_classes="plot-container"):
                        gr.Markdown(
                            "Embeddings encode text into numerical form, allowing models to understand meaning. "
                            "Visualize your text inputs to see how similar they are!\n\n"
                            "Here, I'm using the MiniLM-L6 embedding model that converts each word into a 384-dimensional vector. "
                            "Then I reduce it to 2D using t-SNE for visualization.",
                            elem_classes="markdown-text"
                        )
                        embedding_plot = gr.Plot(label="Text Embedding Visualization")
                        with gr.Column(elem_classes="input-container"):
                            text_input = gr.Textbox(
                                lines=2,
                                placeholder="Enter texts separated by commas (e.g., hello, goodbye, hey, girl)",
                                label="Enter Texts",
                                elem_classes="textbox"
                            )
                            gr.Button(
                                "✨ Generate Embeddings", 
                                variant="primary",
                                size="sm"
                            ).click(
                                lambda texts: generate_embeddings([t.strip() for t in texts.split(",")]),
                                inputs=text_input,
                                outputs=embedding_plot
                            )

                with gr.Tab("🔥 Attention Heatmap"):
                    with gr.Column(elem_classes="plot-container"):
                        gr.Markdown(
                            "Understand how a model focuses on specific words in a sentence. "
                            "This heatmap shows token to token attention strengths.\n\n"
                            "The visualization simulates how attention mechanisms in language models weigh the importance "
                            "between different words in your text.",
                            elem_classes="markdown-text"
                        )
                        attention_plot = gr.Plot(label="Attention Heatmap")
                        with gr.Column(elem_classes="input-container"):
                            text_input = gr.Textbox(
                                lines=2,
                                placeholder="Enter a sentence (e.g., I love sourdough toast with butter and jam.)",
                                label="Enter a Sentence",
                                elem_classes="textbox"
                            )
                            gr.Button(
                                "✨ Generate Heatmap", 
                                variant="primary",
                                size="sm"
                            ).click(
                                generate_attention_heatmap,
                                inputs=text_input,
                                outputs=attention_plot
                            )

    return demo

if __name__ == "__main__":
    app = create_dashboard()
    app.launch()