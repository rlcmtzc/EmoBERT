# EmoBERT
BERT based Model for predicting Emojis. This uses Huggingfaces pretrained RoBERTa model. It gets finetuned on a dataset containing twitter data with 7 different emojis as lables.
![](https://i.imgur.com/AdFWXNO.png)
Figure shows the architecture of EmoBERT.
## Requirments
The code in this repository was tested and created with python version 3.7.  
All needed requirements are listed in `requirements.txt`. To Install them run:
```
pip install -r requirements.txt
```
## Data
The dataset needs to be a pickled pandas dataset consisting of a column `"tweet"` which holds the tweet and a column `"emoji_class"` which holds the numeric representation of the emoji and a column `"emoji"` which holds the unicode representation of the emoji.

## Config for Training/Testing
There is a config file one has to create with all the setup for the model. This looks like the following:
```python
[DEFAULT]
batchsize = 64 # Batchsize for the Dataloading
n_classes = 7 # Number of classes, change this when you want to train on more/less than 7 classes

[TRAIN_MODEL]
modelname_to_save = EmoBERT_base # Model name that gets saved in models/ with epoch attached
test_data = data/emoji_test.pkl # Testdata path, needs to be a pkl file
train_data = data/emoji_train.pkl # Traindata path, needs to be a pkl file
learning_rate = 2e-5 # Learning rate
epochs = 5 # Epochs to train

[TEST_MODEL]
modelname_to_load = models/finetuned_EmoBERT_base_epoch_2.model # model which should be used for testing
test_data = data/emoji_test.pkl # Testdata path, needs to be a pkl file

[PREPROCESSING]
# Here you can add all preprocessing steps needed.
steps: lemmatize_tweets, remove_tweets_with_less_than_two_words
```
Following preprocessing steps are currently supported:  
| Preprocessing Step |  Description   |
|:----:| ---- |
|`remove_stopwords`| Removing all stopwords  |
|`remove_punctuation`| Romove all punctuation|  
|`stem_tweets`| Apply stemming to the data  |
|`lemmatize_tweets`| Apply lemmatizing to the data  |
|`replace_at_with_at`| Replace all freestanding `@` with `at` since freestanding `@` are often used instead of `at` | 
|`remove_extra_whitespace`| All extra whitespaces are removes  |
|`replace_smileys`| All smiley created with punctuation are replaced by a word representing them. | 
|`unify_apostrophes`| All apostrophs get unified to the same one.  |
|`replace_extra_letters`| Some extra unnecessary letters get removed (`Loooooove` -> `Loove`)|  
|`make_lowercase`| Make everything lowercase.|   
|`remove_digits`| Remove all digits  |
|`remove_urls`| Remove all urls (`http` and `https` urls)  |
|`remove_non_english_characters`| Remove all non englisch characters.  |
|`remove_whole_hashtags`| All hashtags are removed (the whole hashtag not only the `#`) | 
|`remove_only_hashtags`| All hashtags are removed (only the `#`) | 
|`remove_freestanding_hashtags`| All freestanding hastags are removed (with a whitespace afterwards). This is usefull after removing non english chars since then all non englisch hashtags result in only a `# `.  |
|`remove_tweets_with_less_than_two_words`| Remove tweets with less then two words.|  
|`apply_random_oversampling`| Apply some random oversampling (Oversample everything to the sample number of the class with the most samples).|  

## Testing the Model
To test the model following command needs to be executed:
`python test_model_roberta.py <path_to_config>`
Add to the config which model you want to test and the path to the testdata.
A pretrained model can be downloaded from [here (Drive)](https://drive.google.com/file/d/1OTY7-aLdjoALKfV1xjfsTuIUxAsFEUKB/view?usp=sharing) (Pretrained on the given [dataset](./data) with the given [config file](./config.config))

## Training the Model
To train (finetune) the model following command needs to be executed:
`python train_model_roberta.py <path_to_config>`
The models get saved in `models/` directory. Further in `results/` the results of the training gets saved as `.json`, this file holds the train/test loss of each epoche, as precision, recall and accuracy metrics. Add all necessary parameters to the config.

## Results
|                  Model                   |  recall   | accuracy  | precision | recall diff | accuracy diff | precision diff |
|:----------------------------------------:|:---------:|:---------:|:---------:|:-----------:|:-------------:|:--------------:|
|                   SVM                    |   56.54   |   58.64   |   54.98   |      -      |       -       |       -        |
|                 EmoBERT                  | **77.72** | **80.86** |   75.82   | **+21.18**  |  **+22.22**   |     +20.84     |
|         EmoBERT w/ oversampling          |   76.28   |   80.09   |   75.96   |   +19.74    |    +21.45     |     +20.98     |
| EmoBERT w/ stowords removed and stemming |   61.42   |   65.00   |   62.74   |    +4.88    |     +6.36     |     +7.76      |
|         EmoBERT w/ Lemmatization         |   76.62   |   80.72   | **77.44** |   +20.08    |    +22.08     |  **+22.46**   |
|           EmoBERT w/ Stemming            |   73.28   |   78.93   |   76.61   |   +16.74    |    +20.29     |     +21.63     |


