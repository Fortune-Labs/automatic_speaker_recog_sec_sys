import os
import librosa
import numpy as np
from sklearn.mixture import GaussianMixture

# Function to train the speaker recognition system
def speaker_recognition_system(dataset_path):

    speakers = []
    mfcc_features = []
    num_gmm_components = 8  # You can adjust the number of Gaussian components
    
    # Step 1: Load the dataset (Assuming the dataset is organized in folders, one folder per speaker)
    for speaker_folder in os.listdir(dataset_path):
        speaker_path = os.path.join(dataset_path, speaker_folder)
        if os.path.isdir(speaker_path):
            speakers.append(speaker_folder)
            features = []

            # Step 2: Extract MFCC features for each speech sample
            for file_name in os.listdir(speaker_path):
                file_path = os.path.join(speaker_path, file_name)
                speech, _ = librosa.load(file_path, sr=None)  # sr=None means no resampling, use the original sampling rate
                mfcc = librosa.feature.mfcc(speech, sr=None, n_mfcc=13)  # n_mfcc is the number of MFCC coefficients
                features.append(mfcc.T)  # Transpose to have shape (time_steps, n_mfcc)

            mfcc_features.append(np.vstack(features))

    # Step 3: Train a GMM for each speaker
    gmm_models = []
    for features in mfcc_features:
        gmm = GaussianMixture(n_components=num_gmm_components, covariance_type='diag')
        gmm.fit(features)
        gmm_models.append(gmm)

    # Save the trained models for future use (optional)
    # You can use the joblib library to save and load the models
    # import joblib
    # joblib.dump((gmm_models, speakers), 'speaker_models.pkl')

    print("Speaker recognition system trained successfully.")
    return gmm_models, speakers

# Function to identify the speaker of a given speech sample
def identify_speaker(speech_sample, gmm_models, speakers):
    
    # Step 4: Extract MFCC features from the speech sample
    mfcc_sample = librosa.feature.mfcc(speech_sample, sr=None, n_mfcc=13).T

    # Step 5: Compute the likelihood of the sample belonging to each speaker using the GMMs
    likelihoods = [gmm.score(mfcc_sample) for gmm in gmm_models]

    # Step 6: Identify the speaker with the highest likelihood
    max_likelihood_idx = np.argmax(likelihoods)
    identified_speaker = speakers[max_likelihood_idx]

    return identified_speaker
