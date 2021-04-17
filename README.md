# AMLSII assignment20_21

## Brief Introduction
This is the code for AMLSII assignment. In the Dataset folder each task have four files which are original dataset, training set, test set and a word to index file for text processing. In the model folder, it has two trained models for two tasks, so you can directly load and test two models without any time-consuming training process.

## How to test the models

Firstly, you should install all the external libraries used by running: <br>
`pip install -r requirements.txt`<br>
I provide two versions of test code. You can go to the **test.ipynb** file and run the code from top to bottom, which will give you the test result. A more convenient way is directly run the python file by:<br>
`python main.py`<br>
Then the programme will automatically run the test for two models. One thing need to point out is that I didn't ignore the warning ouput from programme, thus you may take few more seconds to find the result output.

## How to train the models.

I am also happy if you want to try the complete code including training and testing. However, you need to first download a pre-trained embeeding from :[link](https://mega.nz/file/u4hFAJpK#UeZ5ERYod-SwrekW-qsPSsl-GYwLFQkh06lPTR7K93I). P.S. This is a 1.75Gb large file.<br>
Put the download file into model folder and ensure the path would be **model/datastories.twitter.300d.txt**.<br>
I provide the ipython files in each task folder for my training code. For example, the code for task A would be at this path **A/ipyfile**. Run the ipython code from top to bottom, you will get the result. If you don't want to spend too much time to run the code, you can also check my training records which I also provide at the last part of ipython file.
