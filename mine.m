clear all; close all;

dataDir = './data';

%----------------------------------------------------------------
ads = audioDatastore(dataDir, ...
        IncludeSubfolders=true, ...
        FileExtensions=".wav", ...
        LabelSource="foldernames");

%----------------------------------------------------------------
[adsTrain, adsTest] = splitEachLabel(ads,0.7);

%----------------------------------------------------------------
model = trainModel(adsTrain, adsTest);


%----------------------------------------------------------------

