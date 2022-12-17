from transformers import BlenderbotSmallForCausalLM

# Get the model online
model = BlenderbotSmallForCausalLM.from_pretrained("facebook/blenderbot_small-90M")
# Save the pretrained model
BlenderbotSmallForCausalLM.save_pretrained(model, 'exp/gcnnlm/')