from transformers import TransfoXLTokenizer, TransfoXLModel
import torch

model = TransfoXLModel.from_pretrained("transfo-xl-wt103")
model.save_pretrained("/exp/pytorch_transformer/")
