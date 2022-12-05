from transformers import TransfoXLTokenizer, TransfoXLModel
import torch

model = TransfoXLModel.from_pretrained("transfo-xl-wt103")
traced_model = torch.jit.trace(model)
torch.jit.save(traced_model, "~/exp/pytorch_transformer/")
