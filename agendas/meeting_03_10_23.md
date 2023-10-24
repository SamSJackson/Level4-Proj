# Meeting : 03/10/23

* Time: 15:00-15:30
* Location: SAWB 331
----------

* Project: Watermarking in Machine-Generated Text
* Student: Samuel Jackson
* Student ID: 2520998J
* Supervisor: Dr. Jake Lever
----------

### Agenda

- How Paraphrasers are Trained [5 mins]
- Attempts at GPT2 & LLaMa-2 [5 mins]
- How to generate content [10 mins]
- My plan for next week [5-10 mins]
- [ONLY IF READ KIRCHENBAUER PAPER] Discussion on watermarking technique [5 mins]

### Progress

- Tried out various GPT-2 models (untuned, different parameter numbers).
- Attempted implementing Kirchenbaueur's method.
- Researched & summarised basics of paraphrase training and datasets.
- Setup computer/environments for development.

### Questions

- To generate content, should I prepare prompts/find a prompt dataset?
- Is it necessary to have my own (code) implementation of the watermark method? 
- If I wanted to make improvements to the watermarking, should I leave this as an "if time" objective? 
  - Similarly, would I be able to introduce improvements/alterations without making the changes a focus of the paper?

#### Research Questions
- Does recursive paraphrasing evade the Stanford-Distortion watermarking method?
- Is the degradation of text perplexity linear through paraphrasing?
- Impact of short-context paraphraser vs long-context paraphraser?
- Feasibility of use in academic context?
- Is there a reasonable "minimum" text length for watermarking/paraphrasing? 
- Impact of watermarking given different generation methods (greedy vs beam vs sampling)
- Universal watermark - consequence of watermark

### Meeting Minutes

- Same vocabulary required for generation and detection 
  - Differing vocabularies can have impact on results
- Perhaps streamlit app/implementation of watermark pdf display.
- Types of prompts - should be completion-based as opposed to Q/A
- Perhaps look into document dataset
- Implement my own version of the watermark
- Can just do surface-level improvement of watermark
- The idea of paraphrasing 
  - If so good at paraphrasing then you would just use the paraphrase model as generative model
  - Small paraphraser vs large paraphrasers. Lightweight paraphraser vs heavyweight.
  - Direct paraphrasers may not be directly useful but it is most-likely fine-tuned off of a less paraphrasing model

- Use smaller model (distil vs gpt2-large) at first, save time.

- Treated as a binary classification problem 
  - Can I use z-score to distinguish between watermarked and not-watermarked
- Perhaps try and reproduce some results

- NO MEETING NEXT WEEK 
### Summary

- No meeting next week - next meeting is 17/10/23
- Different types of paraphrasing models is important for this purpose.
- Important to consider how prompt is provided to model - this impacts generation.
- PDF display would be cool.
- Write my own class extension for watermark model.
- Same vocabulary/tokenizer used for model generation is required to detection.

### Next Meeting Goals | Two weeks to complete this
- Implement watermark model which is adaptable.
- Implement generation method
- Find dataset to generate data from - just prompts required.


