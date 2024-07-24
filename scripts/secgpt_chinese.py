from transformers import pipeline

# Create the pipeline
pipe = pipeline("text-generation", model="clouditera/secgpt", trust_remote_code=True)

# Define a prompt
prompt = "Once upon a time"

# Generate text
output = pipe(prompt)

# Print the output
print(output)
