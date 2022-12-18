from transformers import AutoModel, PreTrainedModel, AutoTokenizer, PreTrainedTokenizer

# Get the model online
model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
# Save the pretrained model
PreTrainedModel.save_pretrained(model, 'exp/gcnnlm/')

# Get the tokenizer online
tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
# Save the pretrained tokenizer
PreTrainedTokenizer.save_pretrained(tokenizer, 'exp/gcnnlm/tokenizer/')