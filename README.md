# Speech Emotion Recognition using machine learning
--------------
## Overview:
This project is completely based on  machine learning and deep learning where we train the models with RAVDESS Dataset which consists of audio files which are labelled with basic emotions.
This project is not just about to predict emotion based on the speech. and also to perform some analytical research by applying different machine learning algorithms and neural networks with different architectures.Finally  compare and analyse their results and to get beautiful insights.

## Intro ..
As human beings speech is amongst the most natural way to express ourselves.As emotions play a vital role in communication, the detection and analysis of the same is of vital importance in today’s digital world of remote communication. Emotion detection is a challenging task, because emotions are subjective. There is no common consensus on how to measure or categorize them.

## Dataset Source - RAVDESS

In this project, I use  <a href="https://zenodo.org/record/1188976#.Xl-poCEzZ0w" > RAVDESS</a> dataset to train. 
You can find this dataset in kaggle or click on below link. <br>
https://www.kaggle.com/uwrfkaggler/ravdess-emotional-speech-audio
<br>

2452 audio files, with 12 male speakers and 12 Female speakers, the lexical features (vocabulary) of the utterances are kept constant by speaking only 2 statements of equal lengths in 8 different emotions by all speakers.
This dataset was chosen because it consists of speech and song files classified by 247 untrained Americans to eight different emotions at two intensity levels: Calm, Happy, Sad, Angry, Fearful, Disgust, and Surprise, along with a baseline of Neutral for each actor.

<i>
   **protip** : if you are using google colabs. Use kaggle API to extract data from kaggle with super fast and with super ease :) </i>
 <h4>Pre-requisites : </h4>
    <p>python-3.7+</p>
    <p>librosa</p>
    <p>numpy</p>
    <p>sklearn</p>
    <p>soundfile</p>
   
 ## Data preprocessing
 The heart of this project lies in preprocessing audio files. If you are able to do it . 70 % of project is already done.
 We take benefit of two packages which makes our task easier. 
  - ### LibROSA - for processing and extracting features from the audio file.
  - ### soundfile - to read and write audio files in the storage.
  
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
 
 
 Below is the code snippet to extract features from each file.
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

## Project Details
   
The models which were discussed in the repository are MLP,SVM,Decision Tree,CNN,Random Forest and neural networks of mlp and CNN with different architectures.
   - ### utils.py                - Contains extraction of features,loading dataset functions
   - ### loading_data.py         - Contains dataset loading,splitting data
   - ### mlpclassifier.py        - Contains mlp model code
   - ### Using_ml_algorithms.py  - Contains SVM,randomforest,Decision tree Models.
   - ### CNN_speechemotion.ipynb - Consists of CNN-1d model
   
   
   
<b>NOTE :</b> Remaining .ipynb files were same as above files but shared from google colab.
   
 
 
