# Interactive LLM Explainability Dashboard

**This is an interactive dashboard for exploring and visualizing some internals of language models!**  
I put two main features in this repo:

1. **Text Embeddings Visualization**  
   - Using a SentenceTransformer (all-MiniLM-L6-v2) to generate 384 dimensional text embeddings
   - Then reducing these embeddings to 2D using t-SNE and display them with Plotly
   - The visualization helps you understand how similar or different certain words are
<img width="694" alt="Screenshot 2025-01-29 at 12 52 55 AM" src="https://github.com/user-attachments/assets/ff7dac61-f9b3-4ce0-bc32-bcd64035f227" />

2. **BERT Attention Heatmap**  
   - Loading a pretrained BERT model (bert-base-uncased) with `output_attentions=True`
   - for any given sentence, I extract and average the final layer’s attention weights across all heads
   - then produce a Plotly heatmap (styled beautifully pink) showing how each token attends to every other token- helps you see what words are focused on more
<img width="715" alt="Screenshot 2025-01-29 at 12 54 08 AM" src="https://github.com/user-attachments/assets/885a9c1a-86a6-4541-82f4-f691c6544f51" />

## Installation & Usage

1. **Clone this repo**
   ```bash
   git clone https://github.com/heyanuja/llm-explainability-tool.git
   cd llm-explainability-tool
   ```

2. **Install the dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the dashboard**
   ```bash
   python app.py
   ```
   
## Limitations

- **Text Embedding Visualization**:
  - Requires at least one input text (comma-separated)
  - Works best with 2-30 different text inputs for optimal t-SNE visualization
  - Empty or whitespace-only inputs are filtered out automatically

- **BERT Attention Heatmap**:
  - Limited to 512 tokens maximum (BERT's limitation)
  - Works best with shorter sentences (5-20 words) for clearer visualization
  - Special tokens ([CLS], [SEP]) are automatically removed from the visualization
  - Subword tokens (starting with ##) are cleaned for readability
