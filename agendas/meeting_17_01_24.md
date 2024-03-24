# Meeting : 17/01/23

* Time: 16:00-16:30
* Location: SAWB 331
----------

* Project: Watermarking in Machine-Generated Text
* Student: Samuel Jackson
* Student ID: 2520998J
* Supervisor: Dr. Jake Lever
----------

### Agenda

- Paraphrase similarity vs document similarity
  - Not happy with current similarity comparison method
- Method to decide whether to delve into my own watermark
- Suggested scatterplot visualisation method

### Progress

- Changed data generation method and model - Mistral 7B now.
  - Producing text that I am much happier with.
- Paraphrasing content is looking good; I will bring in some examples tomorrow.
- Tried out using Wieting P-SP - method of paragraph comparison.
- Tried out scatterplot visualisation - possibly going into diss.
- Focusing on writing now.
- Made 1000 document dataset, not evaluated yet.

### Questions

- While trying to calculate document similarity, does choosing a model which larger context window make sense?
- What are the losses in larger context windows?
- How are pretrained models so susceptible to change from finetuning.

### Meeting Minutes

- Choosing a model with large context window will weight each token differently. Depends on the architecture.
- Stick with P-SP similarity as there is a paper to back it up.
- Pretrained models are designed to be changed by finetuning.


