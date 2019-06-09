# UFC fight analysis

This work-in-progress project consists of two files:

## scraper.ipynb
The sole job of this file is to provide the .csv data source. It scrapes over 200 wikipedia html files, each representing a UFC event, extracting fight data and combines them into one 21-column .csv file. It does no data cleaning.

If github rendering for jupyter notebook is not working, you can always view scraper.ipynb with [nbviewer](https://nbviewer.jupyter.org/github/recjo/jupyter-python-ufc/blob/master/scraper.ipynb)


## ufc.ipynb
This file loads the csv file, cleans the data, and creates visualizations from the data. It is a work in progress. Data cleaning has been finalized. I am currently working on the analysis/visualization portion.

If github rendering for jupyter notebook is not working, you can always view ufc.ipynb with [nbviewer](https://nbviewer.jupyter.org/github/recjo/jupyter-python-ufc/blob/master/ufc.ipynb)
