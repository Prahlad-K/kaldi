from transformers import TransfoXLLMHeadModel
import torch

# Get the model online
model = TransfoXLLMHeadModel.from_pretrained("transfo-xl-wt103")
# Save the pretrained model
TransfoXLLMHeadModel.save_pretrained(model, 'exp/transformer_xl/')