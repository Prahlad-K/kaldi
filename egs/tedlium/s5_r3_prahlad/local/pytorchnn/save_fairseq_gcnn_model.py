from transformers import AutoModelForCausalLM

# Get the model online
model = AutoModelForCausalLM.from_pretrained("facebook/blenderbot_small-90M")
# Save the pretrained model
AutoModelForCausalLM.save_pretrained(model, 'exp/gcnnlm/')