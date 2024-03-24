# Meeting : 28/02/2024

* Time: 16:00-16:30
* Location: SAWB 331
----------

* Project: Watermarking in Machine-Generated Text
* Student: Samuel Jackson
* Student ID: 2520998J
* Supervisor: Dr. Jake Lever
----------

### Agenda

- Questions about transformer/decoder architecture [10 mins]
- Decision on including paraphrase similarity [5 mins]

### Progress

- Some beautiful diagrams/images.
- Finetuned paraphraser to a greater extent.
- Making lots of personal questions.

### Questions

- How does a decoder-only model embed the tokens?
  - Why do we not need an encoder?
- After the decoding layers, do we take the final row of the matrix to softamx and generate strategy?

### Meeting Minutes

- Encoder and decoder is old jargon - confusing.
  - Decoder has word embedding in the model. Just not supplied additional information from  encoder model.
- Final row is taken from the embedded matrix after decoder layers.
