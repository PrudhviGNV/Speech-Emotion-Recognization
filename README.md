# Speech Emotion Recognition using machine learning
[![LICENCE.md](https://img.shields.io/github/license/PrudhviGNV/py-automl)](https://github.com/PrudhviGNV/py-automl/blob/master/LICENCE.md)[![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)](https://GitHub.com/PrudhviGNV)[![Open Source Love svg2](https://badges.frapsoft.com/os/v2/open-source.svg?v=103)](https://github.com/PrudhviGNV/open-source-badges/)
[![Awesome Badges](https://img.shields.io/badge/badges-awesome-green.svg)](https://github.com/PrudhviGNV/badges)<br>
--------------
![python-mini-project-speech-emotion-recognition-1280x720](https://user-images.githubusercontent.com/39909903/91180489-9fe39400-e69c-11ea-9968-9adf6741d595.jpg)

----------------
## Overview:
*  This project is completely based on  machine learning and deep learning where we train the models with RAVDESS Dataset which consists of audio files which are labelled with basic emotions.
* #### This project is not just about to predict emotion based on the speech. and also to perform some analytical research by applying different machine learning algorithms and neural networks with different architectures.Finally  compare and analyse their results and to get beautiful insights.

---------------------

## Intro ..
As human beings speech is amongst the most natural way to express ourselves.As emotions play a vital role in communication, the detection and analysis of the same is of vital importance in today’s digital world of remote communication. Emotion detection is a challenging task, because emotions are subjective. There is no common consensus on how to measure or categorize them.

----------

### check out my [Medium blog](https://medium.com/@prudhvi.gnv/as-human-beings-speech-is-amongst-the-most-natural-way-to-express-ourselves-as-8fc38ebe1c44) for quick intuition and understanding

------------

## Dependencies:
- #### python
- #### librosa
- #### soundfile
- #### numpy
- #### keras
- #### sklearn
- #### pandas

---------------
## Project Details
   
The models which were discussed in the repository are MLP,SVM,Decision Tree,CNN,Random Forest and neural networks of mlp and CNN with different architectures.
   - ### utilities.py                - Contains extraction of features,loading dataset functions
   - ### loading_data.py         - Contains dataset loading,splitting data
   - ### mlp_classifier_for_SER.py        - Contains mlp model code
   - ### SER_using_ML_algorithms.py  - Contains SVM,randomforest,Decision tree Models.
   - ### Speech_Emotion_Recognition_using_CNN.ipynb - Consists of CNN-1d model
   <br>
   
<b>NOTE :</b>  Remaining .ipynb files were same as above files but shared from google colab.
-----------------


## Dataset Source - RAVDESS
<br>

In this project, I use  <a href="https://zenodo.org/record/1188976#.Xl-poCEzZ0w" > RAVDESS</a> dataset to train. 
<br>

![s1](https://user-images.githubusercontent.com/39909903/91179186-0798df80-e69b-11ea-824a-f2f65a7c082a.jpg)


<br>
You can find this dataset in kaggle or click on below link. <br>
https://www.kaggle.com/uwrfkaggler/ravdess-emotional-speech-audio
<br>
<br>


2452 audio files, with 12 male speakers and 12 Female speakers, the lexical features (vocabulary) of the utterances are kept constant by speaking only 2 statements of equal lengths in 8 different emotions by all speakers.
This dataset was chosen because it consists of speech and song files classified by 247 untrained Americans to eight different emotions at two intensity levels: Calm, Happy, Sad, Angry, Fearful, Disgust, and Surprise, along with a baseline of Neutral for each actor.

<i>
   <b>protip <b/>: if you are using google colabs. Use kaggle API to extract data from kaggle with super fast and with super ease :) </i>


<H1>Data preprocessing : </H2>
<p>
 The heart of this project lies in preprocessing audio files. If you are able to do it . 70 % of project is already done.
 We take benefit of two packages which makes our task easier. 
 -  <B>LibROSA <B>- for processing and extracting features from the audio file.
 -  <b> soundfile<B> - to read and write audio files in the storage.
  
The main story in preprocessing audio files is to extract features from them.
  
Features supported:
- MFCC (mfcc)
- Chroma (chroma)
- MEL Spectrogram Frequency (mel)
- Contrast (contrast)
- Tonnetz (tonnetz)


In this project, code related to preprocessing the dataset is written in two functions.
- load_data()
- extract_features()




load_data() is used to traverse every file in a directory and we extract features from them and we prepare input and output data for mapping and feed to machine learning algorithms.
and finally, we split the dataset into 80% training and 20% testing.
<br>

```python
def load_data(test_size=0.2):
  X, y = [], []
  try :
    for file in glob.glob("/content/drive/My Drive/wav/Actor_*/*.wav"):
          # get the base name of the audio file
        basename = os.path.basename(file)
        print(basename)
          # get the emotion label
        emotion = int2emotion[basename.split("-")[2]]
          # we allow only AVAILABLE_EMOTIONS we set
        if emotion not in AVAILABLE_EMOTIONS:
              continue
          # extract speech features
        features = extract_feature(file, mfcc=True, chroma=True, mel=True)
          # add to data
        X.append(features)
        y.append(emotion)
  except :
       pass
    # split the data to training and testing and return it
  return train_test_split(np.array(X), y, test_size=test_size, random_state=7)
  ```
 
 

<br>Below is the code snippet to extract features from each file.

  ```python
  
def extract_feature(file_name, **kwargs):
    """
    Extract feature from audio file `file_name`
        Features supported:
            - MFCC (mfcc)
            - Chroma (chroma)
            - MEL Spectrogram Frequency (mel)
            - Contrast (contrast)
            - Tonnetz (tonnetz)
        e.g:
        `features = extract_feature(path, mel=True, mfcc=True)`
    """
    mfcc = kwargs.get("mfcc")
    chroma = kwargs.get("chroma")
    mel = kwargs.get("mel")
    contrast = kwargs.get("contrast")
    tonnetz = kwargs.get("tonnetz")
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if chroma or contrast:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            result = np.hstack((result, mel))
        if contrast:
            contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
            result = np.hstack((result, contrast))
        if tonnetz:
            tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
            result = np.hstack((result, tonnetz))
    return result
   
   ```
   Let's drive further into the project ..
   </p>
   
   --------
   
## Training and Analysis:



### Traditional Machine Learning Models:
Performs different traditional algorithms such as -Decision Tree, SVM, Random forest .

 Refer <br>
 - [SER_using_ML_algorithms.py](https://github.com/PrudhviGNV/SpeechEmotionRecognization/blob/master/SER%20using%20ML%20algorithms.py) or 
 - [Speech Emotion Recognition using ML - SVM, DT, Random Forest.ipynb](https://github.com/PrudhviGNV/SpeechEmotionRecognization/blob/master/Speech%20Emotion%20Recognition%20using%20ML%20-%20SVM%2C%20DT%2C%20Random%20Forest.ipynb)
 
 Finds that these algorithms don’t give satisfactory results. So Deep Learning comes into action.
### Deep Learning:
implements classical neural network architecture such as mlp 
Refer <br>
- [mlp classifier for SER.py](https://github.com/PrudhviGNV/SpeechEmotionRecognization/blob/master/mlp%20classifier%20for%20SER.py) or 
- [Speech Emotion Recognition using MLP.ipynb](https://github.com/PrudhviGNV/SpeechEmotionRecognization/blob/master/Speech%20Emotion%20Recognition%20using%20MLP.ipynb) 

Found that Deep learning algorithms like mlp tends to overfit to the data. So the preferred neural network is CNN which is a game changer in many fields and applications.
Wants to perform some analysis to find the best CNN architecture for available dataset.
Here CNN with different architectures is trained against the dataset and the accuracy is recorded.
Here every architecture has same configuration and is trained to 500 epochs.

Refer <br>
- [Speech Emotion Recognition using CNN.ipynb](https://github.com/PrudhviGNV/SpeechEmotionRecognization/blob/master/Speech%20Emotion%20Recognition%20using%20CNN.ipynb)

### Visualization.
for better understanding about data and also for  visualizing waveform and spectogram of audio files.
Refer <br>
- [emotion_spectogram_CNN_2D.ipynb](https://github.com/PrudhviGNV/SpeechEmotionRecognization/blob/master/emotion_spectogram_CNN_2D.ipynb)


----------------
---------------

## Conclusion and Analysis : 
 - #### Neural networks performs better than traditional classical machine learning models in maximun cases ( by compare metrics) 
 - #### Since Deep learning models are data hunger .. They tend overfit the training data.  (if we keep on training the model. we get 95% +  accuracy :) )
 - #### CNN architectures performs better than traditional neural network architectures. (cnn in most cases perform better than mlp under same configuration)
 - #### CNN with different architectures with same configuration , with same learning rate, with same number of epochs  also have vast difference in the accuracy (from [Speech_Emotion_Recognition_using_CNN.ipynb](https://github.com/PrudhviGNV/SpeechEmotionRecognization/blob/master/Speech%20Emotion%20Recognition%20using%20CNN.ipynb) )

   
 
 
 -------------
 --------------
 
## Hope you like this project :)

-----------
## LICENSE:
[MIT](https://github.com/PrudhviGNV/SpeechEmotionRecognization/blob/master/LICENSE)

 --------
## Contact:
<a href="https://www.linkedin.com/in/prudhvignv"><img src="https://github.com/PrudhviGNV/PrudhviGNV/blob/master/logos/linkedin.png" width="40" /> </a>  <a href="https://github.com/PrudhviGNV"><img src="https://github.com/PrudhviGNV/PrudhviGNV/blob/master/logos/github-logo.png" width="40" /> </a>  <a href="https://www.facebook.com/prudhvi.gnv/"><img src="https://github.com/PrudhviGNV/PrudhviGNV/blob/master/logos/facebook.png" width="40" /> </a>  <a href="mailto:prudhvi.gnv@gmail.com"><img src="https://github.com/PrudhviGNV/PrudhviGNV/blob/master/logos/google-plus.png" width="40" /> </a>  <a
 href="https://www.instagram.com/prudhvi_gnv"><img src="https://github.com/PrudhviGNV/PrudhviGNV/blob/master/logos/instagram.png" width="40" /> </a><a href="https://prudhvignv.github.io"><img src="https://github.com/PrudhviGNV/PrudhviGNV/blob/master/logos/home.png" width="40" /></a>
 
 



 
