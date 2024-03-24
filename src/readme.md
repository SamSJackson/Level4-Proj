# Readme

This dissertation is a research-focused project. Consequently, there is a significant amount of data required to do our computations.

As the data is too large to push to Git, most of the data is not provided. However, the data required for reproducing the results in the dissertation are available.

All the analysis found in the project is based off of the data in `data/processed/perplexity/perplexity_498_18_03_2024.csv`

The code, in the `code` folder, are scripts which generate more data for further analysis, beyond the data mentioned above.

## Build instructions

To build and deploy this project, I have provided an **anaconda** environment list.

### Requirements
Anaconda is required for installation. 

Using the `src/requirements.txt` file, install all the necessary software through the command:

`conda create --name wmark-pt --file requirements.txt`

* Python 3.11
* Packages: listed in `src/requirements.txt` 
* Tested on Windows 10

This will create an anaconda environment named `wmark-pt` and allow you to run all the commands.

Admittedly, you will still have to download the large language models to run the scripts. HuggingFace will automatically download and cache the models as you require them.


