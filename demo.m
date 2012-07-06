%% Data setup
clear all
close all

load toy_2d;
train_features = [pos_feat; neg_feat];
train_features = [train_features, ones(size(train_features,1),1)];
train_labels = [ones(size(pos_feat,1),1); -ones(size(neg_feat,1),1)];

% ************************************************************************
% object = 'car';
% dataset = 'PASCAL2007';
% featureFolder = '../borrow_semisup/';   % folder for the feature file
% datasetMetaFolder = '../borrow_semisup/'; % folder for the dataset meta-data file
% addpath('libsvm-3.12/matlab/');
% addpath('../borrow_semisup/liblinear-1.8/matlab/');
% datasetMetaFile = fullfile(datasetMetaFolder, [dataset, '_meta.mat']);
% % featureFile = fullfile(featureFolder, [dataset, '_llc_2f_' object '.mat']);
% featureFile = 'toy_pascal.mat';
% 
% load(featureFile);
% 
% train_features = [train_features, ones(size(train_features,1),1)];
% test_features = [test_features, ones(size(test_features,1),1)];
% 
% train_labels = train_labels';

%% Parameter setup
C_p = [1000];   % Insert more values for parameter tuning
C_b = [1];
C_svdd = [1];

cnt = 0;
for i = 1 : numel(C_p)
    for j = 1 : numel(C_b)
        for k = 1 : numel(C_svdd)
            cnt = cnt + 1;
            opts{cnt}.C_p = C_p(i);
            opts{cnt}.C_b = C_b(j);
            opts{cnt}.C_svdd = C_svdd(k);
        end
    end
end

org_w = cell(cnt,1);
ball_w = cell(cnt,1);
figure(1);
scatter(neg_feat(:,1),neg_feat(:,2),10,'.');
hold on;
scatter(pos_feat(:,1),pos_feat(:,2),10,'r.');
for i = 1 : cnt
%     ball_w{i} = ball_svm(train_labels, train_features, opts{cnt});
    model = train(double(train_labels), sparse(train_features), ['-s 0 -c ' num2str(opts{i}.C_p) ' -q 1']);
    org_w{i} = model.w';
    px = -8:0.01:5;
    py = -(org_w{i}(1) * px + org_w{i}(3))/org_w{i}(2);
    plot(px, py, '-m', 'LineWidth', 2);
    hold on;
%     org_dec_val{i} = test_features * org_w{i};
%     org_ap(i) = myAP(org_dec_val{i}, test_labels',1);
end