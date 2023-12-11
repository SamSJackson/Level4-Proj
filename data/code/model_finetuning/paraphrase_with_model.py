from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

device = "cuda"
tokenizer = AutoTokenizer.from_pretrained("google/t5-efficient-xl")
model = AutoModelForSeq2SeqLM.from_pretrained("saved/google-t5-efficient-large-nl32-15_000-finetuned", device_map="cuda")

prompt = "'Up!' she screeched."
paragraph = "Harry heard her walking toward the kitchen and then the sound of the frying pan being put on the stove. He rolled onto his back and tried to remember the dream he had been having. It had been a good one. There had been a flying motorcycle in it. He had a funny feeling he'd had the same dream before."
input_text = f"lexical = 100, order = 80 {prompt} <sent> {paragraph} </sent>"

input_ids = tokenizer(input_text, truncation=True, return_tensors='pt', max_length=512, padding='max_length').to(device)

output_tokens = model.generate(input_ids["input_ids"], do_sample=True, top_p=0.75, top_k=None, max_length=512)
print("Generated")
output_text = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]

print(f"INPUT={input_text}\n\nPARAPHRASED={output_text}")
