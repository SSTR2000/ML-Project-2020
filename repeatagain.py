import glob
import os

import librosa  # to extract speech features
import numpy as np
import soundfile  # to read audio file
from sklearn.model_selection import train_test_split  # for splitting training and testing


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
    # to read audio file
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
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel))
        if contrast:
            contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, contrast))
        if tonnetz:
            tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)
            result = np.hstack((result, tonnetz))
    return result


# all emotions on RAVDESS dataset
int2emotion = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

# we allow only these emotions ( feel free to tune this on your need )
AVAILABLE_EMOTIONS = {
    "angry",
    "sad",
    "neutral",
    "happy"
}


def load_data(test_size):
    X, y = [], []
    c = 0
    for file in glob.glob("C:\\Users\\admin\\Desktop\\speech test cases\\Actor_*\\*.wav"):
        # get the base name of the audio file through os path
        basename = os.path.basename(file)
        # get the emotion label
        emotion = int2emotion[basename.split("-")[2]]
        # we allow only AVAILABLE_EMOTIONS we set
        if emotion not in AVAILABLE_EMOTIONS:
            continue
        # extract speech features
        features = extract_feature(file, mfcc=True, chroma=True, mel=True)
        # add to data
        X.append(features)  # will be given as input for both testing and training
        y.append(emotion)  # will be given as training and testing output
    # split the data to training and testing and return it
    return train_test_split(np.array(X), y, test_size=test_size, random_state=7)
    # random state is used to ensure the similarity of training and testing data in each execution


# load RAVDESS dataset, 75% training 25% testing
X_train, X_test, y_train, y_test = load_data(0.25)
# print some details
# number of samples in training data
print("[+] Number of training samples:", X_train.shape[0])
# number of samples in testing data
print("[+] Number of testing samples:", X_test.shape[0])
# number of features used
# this is a vector of features extracted
# using extract_features() function
print("[+] Number of features:", X_train.shape[1])

print("[*] Training the model...")
"""
n_range=range(1,26)
scores={}
scores_list=[]

for n in n_range:
    pca=PCA(n_components=n)
    X_train_1=pca.fit_transform(X_train)
    X_test_1=pca.transform(X_test)
    rfc = RandomForestClassifier(random_state=5)
    rfc.fit(X_train_1, y_train)
    y_pred = rfc.predict(X_test_1)

    #calculating the accuracy
    scores[n]=metrics.accuracy_score(y_test,y_pred)
    scores_list.append(metrics.accuracy_score(y_test,y_pred))

plt.plot(n_range,scores_list)
plt.xlabel('value of n for PCA')
plt.ylabel('Accuracy')
plt.show()
"""
"""
pca=PCA(n_components=150)
X_train_1=pca.fit_transform(X_train)
X_test_1=pca.transform(X_test)

print("Features left after applying PCA",X_train_1.shape[1])

rfc = RandomForestClassifier(random_state=5)
rfc.fit(X_train_1, y_train)
y_pred = rfc.predict(X_test_1)
accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)

print("Accuracy: {:.2f}%".format(accuracy * 100))

output = confusion_matrix(y_test, y_pred)
print("Confusion Matrix.....")
print(output)

output1 = classification_report(y_test, y_pred)
print("Classification Report.....")
print(output1)
"""
