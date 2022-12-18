from transformers import BertModel, BertTokenizer

# Get the model online
model = BertModel.from_pretrained("prajjwal1/bert-tiny")
# Save the pretrained model
BertModel.save_pretrained(model, 'exp/gcnnlm/')

# Get the tokenizer online
tokenizer = BertTokenizer.from_pretrained("prajjwal1/bert-tiny")
# Save the pretrained tokenizer
BertTokenizer.save_pretrained(tokenizer, 'exp/gcnnlm/tokenizer/')