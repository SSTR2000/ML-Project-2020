import glob as glb
import os as OS

import librosa as libs
import numpy as num
import soundfile as snd
from sklearn.model_selection import train_test_split


# from sklearn import svm

# from matplotlib import pyplot as plt
# from sklearn import metrics
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def remove_features(file, **multiargs):
    feature_mfcc = multiargs.get("mfcc")
    feature_chroma = multiargs.get("chroma")
    feature_mel = multiargs.get("mel")
    feature_contrast = multiargs.get("contrast")
    feature_tonnetz = multiargs.get("tonnetz")

    with snd.SoundFile(file) as snd_filename:
        X = snd_filename.read(dtype="float32")
        smp = snd_filename.samplerate
        if feature_chroma or feature_contrast:
            sig_time_freq = num.abs(libs.stft(X))
        outcome = num.array([])
        if feature_mfcc:
            mfccs = num.mean(libs.feature.mfcc(y=X, sr=smp, n_mfcc=40).T, axis=0)
            outcome = num.hstack((outcome, mfccs))
        if feature_chroma:
            chroma = num.mean(libs.feature.chroma_stft(S=sig_time_freq, sr=smp).T, axis=0)
            outcome = num.hstack((outcome, chroma))
        if feature_mel:
            mel = num.mean(libs.feature.melspectrogram(X, sr=smp).T, axis=0)
            outcome = num.hstack((outcome, mel))
        if feature_contrast:
            contrast = num.mean(libs.feature.spectral_contrast(S=sig_time_freq, sr=smp).T, axis=0)
            outcome = num.hstack((outcome, contrast))
        if feature_tonnetz:
            tonnetz = num.mean(libs.feature.tonnetz(y=libs.effects.harmonic(X), sr=smp).T, axis=0)
            outcome = num.hstack((outcome, tonnetz))
    return outcome


available_emotions = {"01": "neutral_emotion", "02": "calm_emotion", "03": "happy_emotion", "04": "sad_emotion",
                      "05": "angry_emotion", "06": "fearful_emotion", "07": "disgust_emotion",
                      "08": "surprised_emotion"}

Considered_emotion = {"angry_emotion", "sad_emotion", "neutral_emotion", "happy_emotion"}


def importing_dataset(test_size):
    X_feature, y_emotion = [], []
    for file in glb.glob("C:\\Users\\admin\\Desktop\\speech test cases\\Actor_*\\*.wav"):

        name = OS.path.basename(file)

        emotion_found = available_emotions[name.split("-")[2]]

        if emotion_found not in Considered_emotion:
            continue

        feature_obtained = remove_features(file, mfcc=True, chroma=True, mel=True)

        X_feature.append(feature_obtained)
        y_emotion.append(emotion_found)

    return train_test_split(num.array(X_feature), y_emotion, test_size=test_size, random_state=7)


X_train_features, X_test_features, y_train_emotion, y_test_emotion = importing_dataset(0.25)

print("[+] Number of training samples:", X_train_features.shape[0])

print("[+] Number of testing samples:", X_test_features.shape[0])

print("[+] Number of features:", X_train_features.shape[1])

print("Shape of training data before applying PCA", X_train_features.shape)
print("Shape of testing data before applying PCA", X_test_features.shape)

from sklearn.preprocessing import StandardScaler

SS = StandardScaler()
X_train_features = SS.fit_transform(X_train_features)
X_test_features = SS.transform(X_test_features)
print("[*] Training the model...")
"""
n_range=range(1,26)
scores={}
scores_list=[]

for n in n_range:
    PC = PCA(n_components=n)
    X_train_1 = PC.fit_transform(X_train_features)
    X_test_1 = PC.transform(X_test_features)

    SVM = svm.SVC(kernel='rbf')  #Linear,rbf,sigmoid,kernel='poly', degree=2
    SVM.fit(X_train_1, y_train_emotion)
    y_prediction = SVM.predict(X_test_1)
    #calculating the accuracy
    scores[n]=metrics.accuracy_score(y_test_emotion,y_prediction)
    scores_list.append(metrics.accuracy_score(y_test_emotion,y_prediction))

plt.plot(n_range,scores_list)
plt.xlabel('value of n for PCA')
plt.ylabel('Accuracy')
plt.show()

"""
"""
PC = PCA(n_components=2)
X_train_1 = PC.fit_transform(X_train_features)
X_test_1 = PC.transform(X_test_features)

print("Shape of training data before applying PCA", X_train_1.shape)
print("Shape of testing data before applying PCA", X_test_1.shape)

SVM = svm.SVC(kernel='rbf')  #Linear,rbf,sigmoid,kernel='poly', degree=2
SVM.fit(X_train_1, y_train_emotion)
y_prediction = SVM.predict(X_test_1)
accuracy = accuracy_score(y_true=y_test_emotion, y_pred=y_prediction)

print("Accuracy: {:.2f}%".format(accuracy * 100))

output = confusion_matrix(y_test_emotion, y_prediction)
print("Confusion Matrix.....")
print(output)

output1 = classification_report(y_test_emotion, y_prediction)
print("Classification Report.....")
print(output1)
"""
