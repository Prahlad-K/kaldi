from transformers import PreTrainedModel, PreTrainedTokenizer

# Get the model online
model = PreTrainedModel.from_pretrained("prajjwal1/bert-small")
# Save the pretrained model
PreTrainedModel.save_pretrained(model, 'exp/gcnnlm/')

# Get the tokenizer online
tokenizer = PreTrainedTokenizer.from_pretrained("prajjwal1/bert-small")
# Save the pretrained tokenizer
PreTrainedTokenizer.save_pretrained(tokenizer, 'exp/gcnnlm/tokenizer/')