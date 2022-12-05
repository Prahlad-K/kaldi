from transformers import TransfoXLTokenizer, TransfoXLModel
import torch

tokenizer = TransfoXLTokenizer.from_pretrained("transfo-xl-wt103")
model = TransfoXLModel.from_pretrained("transfo-xl-wt103", torchscript=True)


# Tokenizing input text
tokenized_text  = tokenizer("Hello, my dog is cute", return_tensors="pt")

# Masking one of the input tokens
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

# Creating a dummy input
tokens_tensor = torch.tensor([indexed_tokens])

# Creating the trace
traced_model = torch.jit.trace(model, tokens_tensor)
torch.jit.save(traced_model, "~/exp/pytorch_transformer/transformer_xl.pt")