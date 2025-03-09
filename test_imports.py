print("Testing imports...")

try:
    import torch
    print("PyTorch imported successfully:", torch.__version__)
    
    from sentence_transformers import SentenceTransformer
    print("SentenceTransformer imported successfully")
    
    import transformers
    print("Transformers imported successfully:", transformers.__version__)
    
    print("All imports successful!")
except Exception as e:
    print(f"Error: {e}")