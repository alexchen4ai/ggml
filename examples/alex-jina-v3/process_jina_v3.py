from jina_xlm_roberta.modeling_lora import XLMRobertaLoRA
from jina_xlm_roberta.configuration_xlm_roberta import XLMRobertaFlashConfig
import warnings
warnings.filterwarnings("ignore")

## The essential code is inside the XLMRobertaLora class

# Initialize the model, pay attention to the code_revision parameter
config = XLMRobertaFlashConfig.from_pretrained("jinaai/jina-embeddings-v3")
model = XLMRobertaLoRA.from_pretrained("jinaai/jina-embeddings-v3", 
                                       config=config)


texts = [
    "Follow the white rabbit.",  # English
    "Sigue al conejo blanco.",  # Spanish
    "Suis le lapin blanc.",  # French
    "跟着白兔走。",  # Chinese
    "اتبع الأرنب الأبيض.",  # Arabic
    "Folge dem weißen Kaninchen.",  # German
]

# When calling the `encode` function, you can choose a `task` based on the use case:
# 'retrieval.query', 'retrieval.passage', 'separation', 'classification', 'text-matching'
# Alternatively, you can choose not to pass a `task`, and no specific LoRA adapter will be used.
embeddings = model.encode(texts, task="text-matching")

# Compute similarities
print(embeddings[0] @ embeddings[1].T)


## We can start with roberta encoder layers 0, ane make sure this works.
layer_0 = model.roberta.encoder.layers[0]

# set some dummy input, let's focus on the layer_0
import torch
batch_size, seq_len, hidden_size = 2, 16, 1024
hidden_states = torch.randn(batch_size, seq_len, hidden_size)

# Get model's dtype
param = next(model.parameters())
model_dtype = param.dtype
print(f"Model dtype: {model_dtype}")

# Convert hidden_states to the same data type and device as the model
hidden_states = hidden_states.to(device=model.device, dtype=model_dtype)

# We need to provide mixer_kwargs as it's expected in the block.py implementation
mixer_kwargs = {}  # Empty dict to avoid the AttributeError

# Run layer with the required mixer_kwargs parameter
output = layer_0(hidden_states, mixer_kwargs=mixer_kwargs)
print(f"Input shape: {hidden_states.shape}, Output shape: {output.shape}")