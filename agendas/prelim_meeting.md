# Meeting : 20/09/2023

* Time: 13:00-13:30PM
* Location: 331 SAWB
----------

* Project: Watermarking in Machine-Generated Text
* Student: Samuel Jackson
* Student ID: 2520998J
* Supervisor: Dr. Jake Lever
----------

### Agenda

- Talk about overarching goal or feasible direction [10 mins]
- Display & discuss timeline [10 mins]
- Required computation [5 mins]
- Weekly meeting time and possible support that can be provided [5 mins]

### Questions

- Is space possible on Glasgow Cluster? Google Colab is also feasible for me. Some papers train on LLaMa which is too big. 
- I would like to focus on academic impact - essay watermarking. Is this okay? As opposed to general purpose papers previously produced.
- Would it be reasonable to email some of the researchers in this topic and ask for opinions on research directions?
- I plan to focus on modifying an existing technique and give deeper statistics on essays in particular.
- I also intend to include basic attacks to defeat watermarking and statistics around this - detection after attack.
- Do you want to me send you any of the papers so far?


- [DetectGPT paper](https://arxiv.org/pdf/2301.11305.pdf) and [UC Santa Barbara](https://arxiv.org/pdf/2306.17439.pdf) both feel that paraphrasing attacks are an area requiring more research. Would this be a feasible direction? 
### Research Questions | Motivation

#### These are questions that are to motivate and to be answered in this paper.

- Can watermarking Large Language Models help aid efforts to maintain academic integrity?
- Can current detection techniques correctly identify AI-generated essays?
- Are watermarks protected against watermark evasion techniques?

### Meeting Minutes

- Arbitrary choices can lead to research questions.
- Keep on reading papers, select some of them to use as visible paper inspiration.
- Between Paraphraser or Academic integrity [KEY PORTION]
- Reproducable functions - don't have chunky notebooks, modularise portions.
- Perplexity could be used as degradation measure.
- Perplexity measured on separate model from the model used to generate watermarked text.
- LLM structural differences - can this be considered in detection evaluation?
- Write a note summarising each paper that I read so that I can quickly refer back to them.
- Meetings should be on Tuesday weekly - time to be decided later.
- First couple weeks should be reading and prepping a solid-base of research and goals.
  - Shouldn't write any code until week 4 or 5. 

### Summary

- Research direction is either:
  - Defense against Paraphrasing Attacks
    - Two separate papers have proposed this as a further research topic.
    - Defense against recursive paraphrasing has not been evaluated, yet.
  - Evaluation of Watermarking Method against GPT-Produced Academic Essays
    - Any essay produced by an LLM.
    - Watermarking method undecided at the moment.

### Next Meeting Goals

- If not decided on direction formally propose both ideas, with pros and cons.

