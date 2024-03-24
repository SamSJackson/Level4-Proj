# Meeting : 28/11/23

* Time: 14:00-14:30
* Location: SAWB 331 
----------

* Project: Watermarking in Machine-Generated Text
* Student: Samuel Jackson
* Student ID: 2520998J
* Supervisor: Dr. Jake Lever
----------

### Agenda

- Look at the new graphs [10mins]
- Questions [5 mins]

### Progress

- Trained on multiple paraphrasing.
- ROC Curves generated.
- Confusion Matrices of data.
- Looking towards future progression.
- Attempted to make watermark, it was very hard.

### Questions

- The existing paraphraser is trained on ChatGPT parpahrases, if I generated on ChatGPT (instead of GPT2), would I expect less degradation in z-score?
- Can I apply a watermark to paraphrasers?
- Perhaps paraphrase by using the initial watermarked model by request.
    - i.e. "Paraphrase this text: {FORMER-GENERATED TEXT}"

### Meeting Minutes

- Don't expect different degradation also ChatGPT generation would be difficult.
- Watermark to paraphrasers is feasible, just would have to add logit processing differently.
- Cool idea. Should research into this.



