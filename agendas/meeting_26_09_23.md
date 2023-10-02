# Meeting : 26/09/2023

* Time: 14:00 - 14:30
* Location: SAWB 331
----------

* Project: Watermarking in Machine-Generated Text
* Student: Samuel Jackson
* Student ID: 2520998J
* Supervisor: Dr. Jake Lever
----------

### Agenda

- Discuss decision to focus on paraphrasing [5-10 mins]
- Prepared short questions [5 minutes]
- Advice on potential research questions to motivate [5 mins]
- Remaining questions [10 mins]

### Questions

- Do I want to construct research questions to structure my essay?
- Perhaps I should discuss instructional content vs conversational content?
- Different lm, different paraphrasers - "arbitrarily" decide?

#### Research Questions
- Does recursive paraphrasing evade the Stanford-Distortion watermarking method?
- Is the degradation of text perplexity linear through paraphrasing?
- Impact of short-context paraphraser vs long-context paraphraser?
- Feasibility of use in academic context?

### Meeting Minutes

- Impact of short-context paraphraser and long-context is feasible as portion of paper
- Academic context, worth to mention perhaps as motivation, maybe look at accesses
- Battle between users vs tools - at end of ML paper, consider impact of F1 score.

- Implement watermark, try on different models, use or modify extent paraphrasing models.
- Find text used for paraphraser training on Transformers.
- How do they train paraphrasers?

- Keep writing down research questions - even if don't like them anymore
- Layout broad problem - first section explaining core conceps

- Each research question would separate each section into the question partitions.

- Instructional vs conversational as a research question, based on finding datasets that would be clearly labelled.
- Will have to make my own data.

- Do not fine-tune models.

- Llama-2/search huggingface for small language models.
- GPT for all

- Perhaps pre-watermarked text? Not very likely based on some papers.

- See if I can generate LlAma content.
- Outliers in document
- Arbitrary choices can be be large and follow if want.

### Summary

- Try and generate some content - this will be necessary to create my dataset.
- Look into how paraphrasers train and what datasets they use.
- Fine-tuning is not feasible option for my computing power.
- Causal models mainly, like GPT models
- Methods on how to write the dissertation
- Keep research questions as they can help guide thinking process and goals.



