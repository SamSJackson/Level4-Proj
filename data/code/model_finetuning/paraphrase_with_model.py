from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

device = "cuda"
tokenizer = AutoTokenizer.from_pretrained("google/t5-efficient-large-nl32")
model = AutoModelForSeq2SeqLM.from_pretrained("saved/google-t5-efficient-large-nl32-25_000-finetuned", device_map="cuda")

# ------------------
prompt = "October arrived, spreading a damp chill over the grounds and into the castle. Madam Pomfrey"
paragraph = ", the nurse, was kept busy by a sudden spate of colds among the staff and students. Her Pepperup potion worked instantly, though it left the drinker smoking at the ears for several hours afterward. Ginny Weasley, who had been looking pale, was bullied into taking some by Percy. The steam pouring from under her vivid hair gave the impression that her whole head was on fire."
input_text = f"lexical = 60, order = 100 {prompt} <sent> {paragraph} </sent>"

input_ids = tokenizer(input_text, truncation=True, return_tensors='pt', max_length=512, padding='max_length').to(device)

output_tokens = model.generate(input_ids["input_ids"], do_sample=True, top_p=0.75, top_k=None, max_length=512)
output_text = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]

print("With prompt")
print(f"{'-'*40}\n")
print(f"INPUT={input_text}\n")
print(f"{'-'*40}\n")
print(f"SENTENCE TO PARAPHRASE={paragraph}\n")
print(f"{'-'*40}\n")
print(f"PARAPHRASED={output_text}\n")

prompt = ""
paragraph = "October arrived, spreading a damp chill over the grounds and into the castle. Madam Pomfrey, the nurse, was kept busy by a sudden spate of colds among the staff and students. Her Pepperup potion worked instantly, though it left the drinker smoking at the ears for several hours afterward. Ginny Weasley, who had been looking pale, was bullied into taking some by Percy. The steam pouring from under her vivid hair gave the impression that her whole head was on fire."
input_text = f"lexical = 60, order = 100 {prompt} <sent> {paragraph} </sent>"

input_ids = tokenizer(input_text, truncation=True, return_tensors='pt', max_length=512, padding='max_length').to(device)

output_tokens = model.generate(input_ids["input_ids"], do_sample=True, top_p=0.75, top_k=None, max_length=512)
output_text = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]

print("NO PROMPT")
print(f"{'-'*40}\n")
print(f"INPUT={input_text}\n")
print(f"{'-'*40}\n")
print(f"SENTENCE TO PARAPHRASE={paragraph}\n")
print(f"{'-'*40}\n")
print(f"PARAPHRASED={output_text}\n")
