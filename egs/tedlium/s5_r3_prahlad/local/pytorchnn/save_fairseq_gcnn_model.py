from transformers import BertModel

# Get the model online
model = BertModel.from_pretrained("bert-base-uncased")
# Save the pretrained model
BertModel.save_pretrained(model, 'exp/gcnnlm/')