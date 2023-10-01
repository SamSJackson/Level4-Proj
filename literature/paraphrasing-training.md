# Overview of Paraphrasing

## Modern Techniques
- Most paraphrasing models are based off of the transformer model.
- On top of this, paraphrasing is mostly a consequence of transfer-learning.
- Models like T5 and ChatGPT have been used for fine-tuning. 

## Training Sets
- Some papers have referenced training through novel-translation, such as [DIPPER](https://arxiv.org/pdf/2303.13408.pdf)
    - Referenced dataset is [Par3](https://github.com/katherinethai/par3). Dataset of books which have been translated from non-english to english
- Some papers reference towards back and forwards translation using google translate.
  - English to French and then back to English.
- Major paraphrasers/most used ones are using datasets like [Microsoft Research Paraphrase Corpus, 5801 Pairs](https://www.google.com/search?q=microsoft+research+paraphrase+corpus&client=firefox-b-d&sca_esv=569871516&sxsrf=AM9HkKnhvhVSbkka29OAp_bhtyFO1-9oVg%3A1696178849102&ei=oaIZZYL2BZ6YhbIPxLmD4AU&ved=0ahUKEwjCyprOptWBAxUeTEEAHcTcAFwQ4dUDCA8&uact=5&oq=microsoft+research+paraphrase+corpus&gs_lp=Egxnd3Mtd2l6LXNlcnAiJG1pY3Jvc29mdCByZXNlYXJjaCBwYXJhcGhyYXNlIGNvcnB1czIFEAAYgAQyBhAAGBYYHkiGK1CcA1jZKnAHeAGQAQGYAYsBoAH5IaoBBTE0LjI3uAEDyAEA-AEBwgIKEAAYRxjWBBiwA8ICBBAjGCfCAgcQIxiKBRgnwgIIEAAYigUYkQLCAg0QLhjHARjRAxiKBRhDwgIHEAAYigUYQ8ICCxAAGIoFGLEDGJECwgITEC4YgwEYxwEYsQMY0QMYigUYQ8ICCBAuGIoFGJECwgINEAAYigUYsQMYgwEYQ8ICCxAuGIAEGLEDGIMBwgIKEAAYigUYsQMYQ8ICBxAjGLECGCfCAg4QABiKBRixAxiDARiRAsICEBAAGIAEGBQYhwIYsQMYgwHCAgsQABiABBixAxiDAcICCBAAGIAEGLEDwgILEC4YgAQYxwEY0QPCAgsQLhiABBjHARivAcICChAAGIAEGBQYhwLCAhAQLhiABBgUGIcCGMcBGNEDwgILEC4YrwEYxwEYgATCAhoQLhiABBjHARjRAxiXBRjcBBjeBBjgBNgBAcICDRAuGA0YgAQYxwEY0QPCAg0QLhgNGK8BGMcBGIAEwgIHEAAYDRiABMICHBAuGA0YgAQYxwEY0QMYlwUY3AQY3gQY4ATYAQHCAhoQLhivARjHARiABBiXBRjcBBjeBBjgBNgBAcICCBAAGIoFGIYDwgIIEAAYFhgeGA_iAwQYACBBiAYBkAYIugYGCAEQARgU&sclient=gws-wiz-serp)
  - These types of datasets are found using similarities in newspapers. 
  - There is also a dataset based on "similar quora questions" - [Quora Question Papers, 400,000 Pairs](https://paperswithcode.com/dataset/quora-question-pairs)

## Desired Paraphraser
- DIPPER - the one used by Krishna in one of the founding papers on paraphrase attacks requires 40GB of RAM. 
  - Not going to happen as I do not have the capacity.
- I could find smaller models and try them out.

## Potential Paraphrasers
- 