from jina_xlm_roberta.modeling_lora import XLMRobertaLoRA
from jina_xlm_roberta.configuration_xlm_roberta import XLMRobertaFlashConfig


# Initialize the model, pay attention to the code_revision parameter
config = XLMRobertaFlashConfig.from_pretrained("jinaai/jina-embeddings-v3", 
                                              trust_remote_code=True)
model = XLMRobertaLoRA.from_pretrained("jinaai/jina-embeddings-v3", 
                                       config=config,
                                #   attn_implementation='eager',
                                  trust_remote_code=True)


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
