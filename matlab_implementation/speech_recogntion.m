function [gmm_models, speakers] = speaker_recognition_system(dataset_path)

    speakers = {};
    mfcc_features = {};
    num_gmm_components = 8;  % You can adjust the number of Gaussian components
    
    % Step 1: Load the dataset (Assuming the dataset is organized in folders, one folder per speaker)
    speaker_folders = dir(dataset_path);
    for i = 1:numel(speaker_folders)
        if speaker_folders(i).isdir && ~strcmp(speaker_folders(i).name, '.') && ~strcmp(speaker_folders(i).name, '..')
            speakers{end+1} = speaker_folders(i).name;
            features = [];
            
            % Step 2: Extract MFCC features for each speech sample
            files = dir(fullfile(dataset_path, speaker_folders(i).name, '*.wav'));
            for j = 1:numel(files)
                file_path = fullfile(dataset_path, speaker_folders(i).name, files(j).name);
                [speech, sr] = audioread(file_path);
                mfcc = melcepst(speech, sr, 'M', 13, floor(3*log(sr)), 160);  % Using Mel-frequency cepstral coefficients (MFCC)
                features = [features; mfcc];  % Concatenate features
            end

            mfcc_features{end+1} = features;
        end
    end

    % Step 3: Train a GMM for each speaker
    gmm_models = {};
    for i = 1:numel(mfcc_features)
        gmm = fitgmdist(mfcc_features{i}, num_gmm_components, 'CovarianceType', 'diagonal');
        gmm_models{end+1} = gmm;
    end

    % Save the trained models for future use (optional)
    % You can use the save and load functions to save and load the models
    % save('speaker_models.mat', 'gmm_models', 'speakers');

    disp('Speaker recognition system trained successfully.');
end

function identified_speaker = identify_speaker(speech_sample, gmm_models, speakers)
    % Step 4: Extract MFCC features from the speech sample
    sr = 16000;  % Adjust the sampling rate if needed
    mfcc_sample = melcepst(speech_sample, sr, 'M', 13, floor(3*log(sr)), 160);  % Using Mel-frequency cepstral coefficients (MFCC)

    % Step 5: Compute the likelihood of the sample belonging to each speaker using the GMMs
    likelihoods = zeros(1, numel(gmm_models));
    for i = 1:numel(gmm_models)
        likelihoods(i) = sum(log(pdf(gmm_models{i}, mfcc_sample)));
    end

    % Step 6: Identify the speaker with the highest likelihood
    [~, max_likelihood_idx] = max(likelihoods);
    identified_speaker = speakers{max_likelihood_idx};
end
