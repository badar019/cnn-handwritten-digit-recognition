%% Handwritten Digit Recognition Using CNN in MATLAB
% Optimized for Generalization and Robustness

clc;
clear;
close all;

%% STEP 1: Load Built-in Digit Dataset

digitDatasetPath = fullfile(matlabroot, ...
    'toolbox','nnet','nndemos', ...
    'nndatasets','DigitDataset');

imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

%% STEP 2: Display Sample Images

figure('Name','Sample Digit Images');

perm = randperm(numel(imds.Files),20);

for i = 1:20
    
    subplot(4,5,i)

    img = readimage(imds,perm(i));

    imshow(img)

    label = imds.Labels(perm(i));

    title(char(label))
end

%% STEP 3: Split Dataset into Training and Testing

[imdsTrain, imdsTest] = splitEachLabel(imds,0.8,'randomized');

%% STEP 4: Data Augmentation

imageAugmenter = imageDataAugmenter( ...
    'RandRotation',[-10 10], ...
    'RandXTranslation',[-3 3], ...
    'RandYTranslation',[-3 3]);

augimdsTrain = augmentedImageDatastore( ...
    [28 28 1], ...
    imdsTrain, ...
    'DataAugmentation',imageAugmenter);

%% STEP 5: CNN Architecture

layers = [

    imageInputLayer([28 28 1])

    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)

    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)

    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer

    dropoutLayer(0.2)

    fullyConnectedLayer(10)

    softmaxLayer

    classificationLayer
];

%% STEP 6: Training Options

options = trainingOptions('adam', ...
    'InitialLearnRate',0.001, ...
    'MaxEpochs',8, ...
    'MiniBatchSize',64, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsTest, ...
    'Verbose',false, ...
    'Plots','none');

%% STEP 7: Train CNN Model

net = trainNetwork(augimdsTrain,layers,options);

%% STEP 8: Predict Test Data

predictedLabels = classify(net,imdsTest);

trueLabels = imdsTest.Labels;

%% STEP 9: Calculate Accuracy

accuracy = sum(predictedLabels == trueLabels) ...
    / numel(trueLabels);

fprintf('\n====================================\n');
fprintf(' Handwritten Digit Recognition\n');
fprintf('====================================\n');
fprintf('Final Test Accuracy = %.2f%%\n', accuracy*100);
fprintf('====================================\n');

%% STEP 10: Confusion Matrix

figure('Name','Confusion Matrix');

confusionchart(trueLabels,predictedLabels);

title('Confusion Matrix');

%% STEP 11: Show Random Sample Predictions

figure('Name','Sample Predictions');

idx = randperm(numel(imdsTest.Files),9);

for i = 1:9

    subplot(3,3,i)

    img = readimage(imdsTest,idx(i));

    imshow(img)

    pred = predictedLabels(idx(i));

    actual = trueLabels(idx(i));

    if pred == actual
        color = 'green';
    else
        color = 'red';
    end

    title( ...
        ['Pred: ' char(pred) ...
        ' | Actual: ' char(actual)], ...
        'Color',color);

end

%% STEP 12: Save Trained Model

save('trainedDigitCNN_Optimized.mat','net');

fprintf('\nModel saved successfully.\n');
fprintf('Project Completed Successfully.\n');