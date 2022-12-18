from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Get the model online
model = GPT2LMHeadModel.from_pretrained("distilgpt2")
# Save the pretrained model
GPT2LMHeadModel.save_pretrained(model, 'exp/gcnnlm/')

# Get the tokenizer online
tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
# Save the pretrained tokenizer
GPT2Tokenizer.save_pretrained(tokenizer, 'exp/gcnnlm/tokenizer/')