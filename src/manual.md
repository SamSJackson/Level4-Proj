# User manual 

## Specifications
Note that this project is computationally intensive.
The following specifications were used to run my programs

```
    GPU: GeForce RTX 3090  | VRAM: 24GB
    CPU: AMD Ryzen 5800X3D | Clock speed: 3401Mhz, 8 cores.
    RAM: 64GB 
```

For running inference tasks, the memory cost is typically 4x the parameter count of a model
    - i.e. 1.5B parameter model would require ~6GB RAM to run inference.

Finetuning or training a model is typically more intensive and requires more RAM.

## Data Creation
### Generation
This folder is the bulk of the code. The code is structured as a pipeline.

In order to avoid wasting time or losing information, data is saved at each stage of the pipeline.

To run the research process outlined in my dissertation, run the `data-pipeline.py` file. Within the file, there is variables that determine:
- No. documents to generate
- Watermark strength
- Paraphrase strength
- No. repeated attacks
- Percentage of word replacement
- Target output directory

### Implementation
Some of the work completed was from code provided by other people.
The `maryland` directory is the Maryland Watermark, with code provided by Kirchenbauer: [github](https://github.com/jwkirchenbauer/lm-watermarking). Our code is slightly different but the logical process remains. Alterations were completed as a consequence of personal preference.

The `wieting` directory is the paraphrase similarity model, with code provided by Wieting [github](https://github.com/jwieting/paraphrastic-representations-at-scale). 
This github does not contain the relevant models because the files are too large but the linked github will provide guidance as to where to find the models.

### Finetuning
Finetuning was completed for our paragraph-based paraphraser.

The technique follows the methods proposed by Krishna: [paper](https://arxiv.org/pdf/2303.13408.pdf).

The dataset that I used is available on HuggingFace: [dataset](https://huggingface.co/datasets/SamSJackson/kpar3-no-ctx)

Similarly, the finetuned model that I created is on HuggingFace: [model](https://huggingface.co/SamSJackson/paraphrase-dipper-no-ctx)

## Data Exploration / Analysis
All data exploration and analysis is completed in Jupyter Notebooks.
These notebooks are all provided in the `notebooks` folder.

`confusion_matrices.ipynb` is a notebook performing exploration of the data through a binary classification of detection.

`plot_roc_curve.ipynb` explores the data through understanding the watermarking strength and quality.

`plot_z_scores.ipynb` provides a visual analysis of the change in z-scores, after attacking phases.