# FRB Project

Contains python files and data about the FRB project I worked on with Professor Petrosian from 2020-2022.

### Notes About Files

Due to the relocation of many files, many of the .py or .ipynb files may no longer run smoothly. Please check for broken links in the importing of any package or any data if you're having trouble running a file.

### Conda Environment

To recreate the Conda environment used to run the Python files in this repo, use the environment.yml file as follows:

<code>conda env update --name (env_name) --file environment.yml </code>

where <code>env_name</code> is the name of the new environment in which you would like to run this code.

Python 3.9.12 was used to run files in this repo.

### Saving Conda Environment

If any changes to the conda environment are made, run <code>conda env export > environment.yml</code> to save the new environment file so that everything run can be recreated.