from transformers import BlenderbotSmallTokenizer, BlenderbotSmallForCausalLM

# Get the model online
model = BlenderbotSmallForCausalLM.from_pretrained("facebook/blenderbot_small-90M")
# Save the pretrained model
BlenderbotSmallForCausalLM.save_pretrained(model, 'exp/gcnnlm/')

# Get the tokenizer online
tokenizer = BlenderbotSmallTokenizer.from_pretrained("facebook/blenderbot_small-90M")
# Save the pretrained tokenizer
BlenderbotSmallTokenizer.save_pretrained(model, 'exp/gcnnlm/tokenizer/')