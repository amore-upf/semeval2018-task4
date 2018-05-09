![AMORE-UPF](logos/logo-AMORE-blue-withtext.png)    &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;      ![UPF](logos/upf-logo.png)

# AMORE-UPF  

Accompanying code for our participating system *AMORE-UPF* at [SemEval 2018 Task 4: Character Identification on Multiparty Dialogues](https://competitions.codalab.org/competitions/17310). 
See the link below for our paper with the system description. 

### Usage
For training, testing, or evaluation, run the `main.py` script with the corresponding parameters (more details to the different phases are given below): 

`python main.py --phase <phase> [-c <config_file>] [--model <model_path>] [--deploy_data <path_to_data>] [--no_cuda] [--no_eval]`

where 
- `phase` can be train or deploy (optionally runs evaluation)
- `config_file` specifies the hyperparameter settings. Is obligatory for training. 
- `model_path` specifies the path to the model. It is obligatory for the deploy phase. 
- `deploy_data` gives the path to the data for which the model has to output predictions (in [CONLL](https://competitions.codalab.org/competitions/17310#learn_the_details-evaluation) format) (phase: deploy)
- `no_eval` applies to the deploy phase. It can be set if you do not want to evaluate the model, but just want to obtain predictions for some input data. If the input data does not contain target entity ids, `no_eval` is set by default.
- `no_cuda`is set to run the system in CPU mode.

### SemEval task data
First please download the SemEval datasets by running the script `_fetch_data.sh` from within the semeval-task4 folder. (Alternatively, you can download the data yourself from [the organizers' github](https://github.com/emorynlp/semeval-2018-task4/tree/master/dat). Store them in the folder data/friends.)


### Running and evaluating the AMORE-UPF model on the SemEval test data

<code>python main.py --deploy_data test --model models/semeval-winning-model/amore-upf [--no_cuda]</code>

This will produce the following output files, saved in the directory 
<tt>models/semeval-winning-model/answers/friends_test_scene/</tt> :

- <tt>amore-upf--ensemble.csv</tt><br/>
*The answer file: It has three columns (called index, prediction, target), <br/>where each row contains the index of the target mention in the test data, the predicted entity id, and the gold entity id to which the mention refers*

- <tt>amore-upf--ensemble_scores.txt</tt> <br/>
*The evaluation results.*

- <tt>amore-upf--ensemble_matrix.csv</tt><br/>
*A confusion matrix.*

- <tt>amore-upf.ini</tt><br/>
*The used config file.*

### Demo

The demo describes how to train, deploy and evaluate a model from scratch using the official trial data of the SemEval task. If you have not used the script `_fetch_data.sh` ([see the section on task data above](#semeval-task-data)) for data download, you first need to get the trial data [here](https://competitions.codalab.org/my/datasets/download/d8e0b7e1-1c4f-4171-93e9-74339e6c759e).

#### Training
`python main.py --phase train -c config_demo.ini [-r] [--no_cuda]`

where the optional parameter 
* `r` is used to activate random sampling of hyperparameters  from intervals specified in the config file. (see <tt>config_demo.ini</tt> for details)
* See above for the description of the other parameters.

The system will produce a subfolder `<year_month>` in the `models` directory, in which it will store several files:
* the config file
* the model file (or files, if run with cross-validation, see parameter `folds` in the config), 
* a `logs` subfolder with the training log (it records the loss, accuracy etc. on the training and validation data for each epoch).

The files will contain a timestamp in their name in the format `<yyyy_mm_dd_hh_mm_ss>`. 

For example, running the command above in April 2018 will train a model with 2-fold cross-validation, and produce something like 
```
.
|__ `models/2018_04/`
|  |  `fixed--2018_04_19_17_58_14.ini`
|  |  `fixed--2018_04_19_17_58_14--fold0.pt`
|  |  `fixed--2018_04_19_17_58_14--fold1.pt`
|  |__`logs/`
|      | `fixed--2018_04_19_17_58_14.log`
|      | `fixed--2018_04_19_17_58_14.ini`
```
The prefix <tt>fixed</tt> means that the model was trained using fixed hyperparameters (since parameter `r` was not set, see above).

##### Using pre-trained word embeddings
Note that the model in this demo initialises the token embeddings randomly. If you want to use  the pre-trained Google News skip-gram word embeddings (as AMORE-UPF does), you first need to download the data. You can do so either by setting the parameter in <tt>GET_GOOGLE_NEWS_EMBEDDINGS</tt> in `_fetch_data.sh`  to true and running the script again. Or you can directly download the vectors from here: 
[GoogleNews-vectors-negative300.bin.gz](https://code.google.com/archive/p/word2vec/).
Put this in the data/ folder. <br/>
In <tt>config_demo.ini</tt>, set the parameter <tt>token emb</tt> to <tt>google_news</tt>.



#### Evaluation
###### Evaluate the model on the trial data
The system was trained using 2-fold cross-validation. So for evaluation on the trial data (on which it was trained), it averages the scores of each fold's models obtained on the respective test split:

`python main.py --phase deploy --deploy_data trial --model models/2018_04/fixed--2018_04_20_11_28_19 [--no_cuda]`

This will produce a subfolder `answers/friends_trial_scene/` in the model subfolder `models/2018_04/`. 
See the [Section above](#running-and-evaluating-the-amore-upf-model-on-the-semeval-test-data) for the description of the files stored therein.

###### Evaluate the model on the test data
`python main.py --phase deploy --deploy_data test --model models/2018_04/fixed--2018_04_20_11_28_19 [--no_cuda]`

See the [Section above](#running-and-evaluating-the-amore-upf-model-on-the-semeval-test-data) for details.


#### Deploying: Run the trained system on an input document 
`python main.py --phase deploy --deploy_data <path_to_data> --model <path_to_model> [--no_cuda] [--no_eval]`

where
* `<path_to_model>` specifies the path and the prefix of the model file, e.g., <br/>
    <tt>models/2018_04/fixed--2018_04_20_11_28_19</tt>
* `<path_to_data>` gives the path to the data file, e.g.,<br/>
    <tt>data/friends/friends.test.scene_delim.conll.nokeys</tt>
* `no_eval` can be set if you do not want to evaluate the model, but just want to obtain predictions for some input data. If the input data does not contain target entity ids, `no_eval`is set by default.

## Citation
We release the source code under the <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0 license</a>, so feel free to share and/or adapt, provided you give appropriate credit.

The system is described in this paper [pdf]: <a target="_blank" href="http://www.coli.uni-saarland.de/~carina/semeval2018-amore_upf-final.pdf">AMORE-UPF at SemEval-2018 Task 4: BiLSTM with Entity Library</a>

```
@inproceedings{ aina-silberer-sorodoc-westera-boleda:2018:SemEval,
    title     = {AMORE-UPF at SemEval-2018 Task 4: BiLSTM with Entity Library},
    author    = {Aina, Laura and Silberer, Carina and Sorodoc, Ionut-Teodor and Westera, Matthijs and Boleda, Gemma},
    booktitle = {Proceedings of the 12th International Workshop on Semantic Evaluation (SemEval-2018)},
    pages     = {(to appear)},
    month     = {June},
    year      = {2018},
    address   = {New Orleans, Louisiana},
    publisher = {Association for Computational Linguistics},
}
```

## Acknowledgements
This project has received funding from the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation programme (grant agreement No 715154), and from the Spanish Ram\'on y Cajal programme (grant RYC-2015-18907). We are grateful to the NVIDIA Corporation for the donation of GPUs used for this research. We are also very grateful to the Pytorch developers. This paper reflects the authors' view only, and the EU is not responsible for any use that may be made of the information it contains.

![(ERC logo)](logos/LOGO-ERC.jpg)      &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;       ![(EU flag)](logos/flag_yellow_low.jpeg)


