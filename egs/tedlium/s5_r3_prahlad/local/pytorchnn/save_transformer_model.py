from transformers import TransfoXLTokenizer, TransfoXLModel
import torch

# Get the model online
model = TransfoXLModel.from_pretrained("transfo-xl-wt103")
# Save the pretrained model
TransfoXLModel.save_pretrained(model, 'exp/transformer_xl/')