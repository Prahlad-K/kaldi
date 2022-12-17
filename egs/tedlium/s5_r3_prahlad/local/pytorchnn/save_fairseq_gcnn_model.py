from transformers import BertLMHeadModel

# Get the model online
model = BertLMHeadModel.from_pretrained("bert-base-cased")
# Save the pretrained model
BertLMHeadModel.save_pretrained(model, 'exp/gcnnlm/')