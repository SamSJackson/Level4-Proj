from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import sys

device = "cuda"

source_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es")
source_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-es", device_map=device)

target_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-es-en")
target_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-es-en", device_map=device)

og_text = "He is confiding in me!"
print(f"Original Text: {og_text}")

en_tokenized = source_tokenizer(og_text, return_tensors='pt').to(device)
en_outputs = source_model.generate(**en_tokenized)

es_decoded = source_tokenizer.batch_decode(en_outputs, skip_special_tokens=True)[0]
print(f"Spanish Translation: {es_decoded}")

es_tokenized = target_tokenizer(es_decoded, return_tensors='pt').to(device)
es_outputs = target_model.generate(**es_tokenized)

es_outputs = target_tokenizer.batch_decode(es_outputs, skip_special_tokens=True)[0]
print(f"Final: {es_outputs}")