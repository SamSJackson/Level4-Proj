# Meeting : 17/10/2023

* Time: 14:00-14:30
* Location: SAWB 331
----------

* Project: Watermarking in Machine-Generated Text
* Student: Samuel Jackson
* Student ID: 2520998J
* Supervisor: Dr. Jake Lever
----------

### Agenda

- Questions [5-10mins]
- How to prepare code/goal for releasing code [5mins]
- Preparation for writing dissertation [5mins]
- Plan for upcoming weeks [5mins]

### Progress

- Attempted some greedy generation on GPT2
  - CPU
    - 29 documents, with sampling, took 6 minutes.
    - 12.4 seconds per doc; assuming linear increase: 3h 20mins for 1000 documents.
  - GPU 
    - 31 documents, with sampling, took 2m 42seconds
    - 5.2 seconds per doc; assuming linear increase: 1h 26mins for 1000 documents
- Probably useful to have these stats amongst my diss.
- Created Generator class which allows document generation with adaptable model - fully adaptable outside required prompt changes.
- Adaptable watermarking class 

### Questions

- In the case of paraphrasing with beam search:
  - Will paraphasing twice have similar results to picking the second most likely option in beam search?

#### Research Questions
- Does recursive paraphrasing evade the Stanford-Distortion watermarking method?
- Is the degradation of text perplexity linear through paraphrasing?
- Impact of short-context paraphraser vs long-context paraphraser?
- Feasibility of use in academic context?
- Is there a reasonable "minimum" text length for watermarking/paraphrasing? 
- Impact of watermarking given different generation methods (greedy vs beam vs sampling)
- NO NEW RESEARCH QUESTIONS THIS WEEK!

### Meeting Minutes

- Notes throughout session

### Summary

- Summary of meeting

### Next Meeting Goals

- I need to research potential paraphrasing models - this has not been done yet.


