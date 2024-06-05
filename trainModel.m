function m = trainModel(adsTrain, adsTest)

%% 1
    [sampleTrain, dsInfo] = read(adsTrain);  
    reset(adsTrain);

    fs = dsInfo.SampleRate;
    windowLength = round(0.03*fs);
    overlapLength = round(0.025*fs);
    afe = audioFeatureExtractor(SampleRate=fs, ...
        Window=hamming(windowLength,"periodic"), OverlapLength=overlapLength, ...
        zerocrossrate=true, shortTimeEnergy=true, pitch=true, mfcc = true);
    
    featureMap = info(afe)
    
    features = [];
    labels = [];
    lengths = [];
    energyThreshold = 0.6;
    zcrThreshold = 0.2;
    while hasdata(adsTrain)
        [audioIn, dsInfo] = read(adsTrain);
    
        feature = extract(afe, audioIn);
        isSpeech = feature(:, featureMap.shortTimeEnergy) > energyThreshold;
        isVoiced = feature(:, featureMap.zerocrossrate) < zcrThreshold;
    
        voicedSpeech = isSpeech & isVoiced;
    
        feature(~voicedSpeech,:) = [];

        feature(:,[featureMap.zerocrossrate,featureMap.shortTimeEnergy]) = [];

        label = repelem(dsInfo.Label,size(feature,1));
        lengths = [lengths;size(feature,1)];
        features = [features;feature];
        
        assignin('base', "test", features);
        labels = [labels,label];
        assignin('base', 'features', features);
    end
    global M; global S;
    M = mean(features,1);
    S = std(features, [], 1);
    features = (features-M)./S; % Pitch 和 MFCC 的比例不同。这将使分类器产生偏差。通过减去均值并除以标准差来归一化特征。
    
    %----------------------------------------------------------------

    %Train Model(s)

    %KNN Classifier
    trainedClassifier = fitcknn(features, labels, ...
        Distance="euclidean", ...
        NumNeighbors=5, ...
        DistanceWeight="squaredinverse", ...
        Standardize=false, ...
        ClassNames=unique(labels));
    
    %Naiive Bayes
    bayesModel = fitcnb(features,labels,'ClassNames',unique(labels));
    assignin('base','bayesModel',bayesModel);
    %----------------------------------------------------------------

    k = 5;
    group = labels;
    c = cvpartition(group, KFold=k);
    partitionedModel = crossval(trainedClassifier, CVPartition=c);
    validationAccuracy = 1 - kfoldLoss(partitionedModel, LossFun="ClassifError");
    fprintf('\nValidation accuracy = %.2f%%\n', validationAccuracy*100);
    
    %----------------------------------------------------------------

    %Testing Classifier
    
    features = [];
    labels = [];
    numVectorsPerFile = [];
    while hasdata(adsTest)
        [audioIn, dsInfo] = read(adsTest);
    
        feature = extract(afe, audioIn);
    
        isSpeech = feature(:,featureMap.shortTimeEnergy) > energyThreshold;
        isVoiced = feature(:,featureMap.zerocrossrate) < zcrThreshold;
    
        voicedSpeech = isSpeech & isVoiced;
    
        feature(~voicedSpeech,:) = [];
        numVec = size(feature,1); % 该样本剩余的音频帧数量
        feature(:,[featureMap.zerocrossrate,featureMap.shortTimeEnergy]) = [];
    
        label = repelem(dsInfo.Label, numVec);
    
        numVectorsPerFile = [numVectorsPerFile, numVec]; % 每个样本的剩余音频帧数量，行向量
    
        features = [features;feature];
        labels = [labels, label];
    end
    features = (features-M)./S;
    
    prediction = predict(trainedClassifier,features); %通过调用 predict来预测每一帧的标签（扬声器）。trainedClassifier
    prediction = categorical(string(prediction));
    
    r2 = prediction(1:numel(adsTest.Files));
    idx = 1;
    for ii = 1:numel(adsTest.Files)
        r2(ii) = mode(prediction(idx:idx+numVectorsPerFile(ii)-1));
        idx = idx + numVectorsPerFile(ii);
    end
    
    figure(Units="normalized",Position=[0.4 0.4 0.4 0.4])
    confusionchart(adsTest.Labels,r2,title="KNN分类混淆矩阵", ...
    ColumnSummary="column-normalized",RowSummary="row-normalized");
    
    m = trainedClassifier;

%----------------------------------------------------------------

    predlabels = predict(bayesModel,features);

    r3 = predlabels(1:numel(adsTest.Files));
    idx = 1;
    for ii = 1:numel(adsTest.Files)
        r3(ii) = mode(predlabels(idx:idx+numVectorsPerFile(ii)-1));
        idx = idx + numVectorsPerFile(ii);
    end

    %table(adsTest.Labels,r3,'VariableNames',...
    %{'TrueLabel','PredictedLabel'})
    figure(Units="normalized",Position=[0.4 0.4 0.4 0.4])
    confusionchart(adsTest.Labels,r3,title="朴素贝叶斯分类混淆矩阵", ...
    ColumnSummary="column-normalized",RowSummary="row-normalized");
end