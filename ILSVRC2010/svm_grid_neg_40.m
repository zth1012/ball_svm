close all
clear all

%  pos_IL_ID = 779;     class = 'Airship';
% pos_IL_ID = 449;        class = 'Lion';
% pos_IL_ID = 761;    class = 'Monitor';
% pos_IL_ID = 811;    class = 'Armchair';
% pos_IL_ID = 818;    class = 'Diningtable';

% pos_IL_ID = 563;    class = 'Keyboard';
% pos_IL_ID = 837;    class = 'Church';
% pos_IL_ID = 22;     class = 'Banana';
pos_IL_ID = 106;    class = 'Sunflower';

load 'valid_data.mat';

if ispc
    path(path,'../3rd-party/liblinear-1.8/windows');
else
    %     path(path,'../3rd-party/liblinear-1.5/matlab');
    path(path,'../3rd-party/liblinear-1.8/matlab');
end

NegTrainPerCat = 40;
NumPosTrain = 500;
train_folder = 'train_data/';
model_folder = 'svm_models/';
% MaxIter = 20;
% BatchSize = 5*999;
% NegSVMWeight = NumPosTrain/BatchSize;
Cost =  [0.1 1 10];
AP = zeros(numel(Cost),1);
Model = cell(numel(Cost),1);
DecValues = cell(numel(Cost),1);
TrainFileName = [train_folder class '_neg_' num2str(NegTrainPerCat) '.mat'];
load(TrainFileName);

for K = 1 : numel(Cost)
    SVMOpts = ['-c ', num2str(Cost(K))];   
    fprintf('Training: %d/%d\n', K, numel(Cost));
    Model{K} = train([PosTrainLabels; NegTrainLabels], [PosTrainFeatures; NegTrainFeatures], SVMOpts);
    
%     [Model{K}, IsConverge] = mineHardNegatives(PosTrainLabels, PosTrainFeatures, NegTrainLabels, NegTrainFeatures, ...
%         MaxIter, BatchSize, 'liblinear', SVMOpts);
    
    % Test on validation data
    
    [PredIgnore, AccIgnore, DecValues{K}] = predict(ValidLabels, ...
        ValidFeatures, ...
        Model{K});
    
    AP(K) = eval_ap(DecValues{K}, ValidLabels, PosImageNetID);
    
    fprintf('Cost = %f, AP = %f\n', Cost(K), AP(K));
end

for K = 1 : numel(Cost)
    fprintf('Cost = %f, AP = %f\n', Cost(K), AP(K));
end

[MaxAP, MaxInd] = max(AP(:));
fprintf('The maximum AP achieved: %f, C = %f\n', MaxAP, Cost(MaxInd));
BestModel = Model{MaxInd};

% save(['svm_param_search_ap_neg_' num2str(NegTrainPerCat) '.mat'], 'Cost', 'AP');
save([model_folder class '_best_model_neg_' num2str(NegTrainPerCat) '.mat'], 'BestModel');


