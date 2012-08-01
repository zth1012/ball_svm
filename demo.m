%% Data setup
clear all
close all

load toy_2d;
train_features = [pos_feat; neg_feat];
train_features = [train_features,ones(size(train_features,1),1)];
train_labels = [ones(size(pos_feat,1),1); -ones(size(neg_feat,1),1)];

% ************************************************************************
% object = 'car';
% dataset = 'PASCAL2007';
% featureFolder = '../borrow_semisup/';   % folder for the feature file
% datasetMetaFolder = '../borrow_semisup/'; % folder for the dataset meta-data file
% addpath('libsvm-3.12/matlab/');
addpath('../borrow_semisup/liblinear-1.8/matlab/');
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
C_p = [40];   % Insert more values for parameter tuning
C_n = [10];
C_svdd = [0.01];
% K = [1 2 5 10 20];
K = 10;

cnt = 0;
for i = 1 : numel(C_p)
    for j = 1 : numel(C_n)
        for k = 1 : numel(K)
            for m = 1 : numel(C_svdd)
                cnt = cnt + 1;
                opts{cnt}.C_p = C_p(i);
                opts{cnt}.C_n = C_n(j);
                opts{cnt}.C_svdd = C_svdd(m);
                opts{cnt}.eta_c = 0.00001;
                opts{cnt}.eta_r = 0.001;
                opts{cnt}.eta_w = 0.000001; 
                opts{cnt}.max_iter_c = 50;
                opts{cnt}.max_iter_r = 50;
                opts{cnt}.max_iter_w = 1000;
                opts{cnt}.global_max_iter = 5;
                opts{cnt}.K = K(k);
            end
        end
    end
end

org_w = cell(cnt,1);
ball_w = cell(cnt,1);
neg_balls = cell(cnt,1);

for i = 1 : cnt
    [ball_w{i}, neg_balls{i}, kmeans_w{i}] = ball_svm(train_labels, train_features, opts{i});
    model = train(double(train_labels), sparse(train_features), ['-s 0 -c ' num2str(opts{i}.C_p) ' -q 1']);
    org_w{i} = model.w';
    
    [px_proj_org, ~] = arrayfun(@(x,y) project_point_to_line(org_w{i}, x, y), train_features(:,1), train_features(:,2));
    [px_proj_ball, ~] = arrayfun(@(x,y) project_point_to_line(ball_w{i}, x, y), train_features(:,1), train_features(:,2));
    [px_proj_kmeans, ~] = arrayfun(@(x,y) project_point_to_line(kmeans_w{i}, x, y), train_features(:,1), train_features(:,2));
    px_org = min(px_proj_org):0.01:max(px_proj_org);
    px_ball = min(px_proj_ball):0.01:max(px_proj_ball);
    px_kmeans = min(px_proj_kmeans):0.01:max(px_proj_kmeans);
    %     [min_x, min_idx] = min(train_features(:,1));
    %     [max_x, max_idx] = max(train_features(:,1));
    %     min_y = train_features(min_idx,2);
    %     max_y = train_features(max_idx,2);
    %     [proj_min_x_ball, ~] = project_point_to_line(ball_w{i}, min_x, min_y);
    %     [proj_max_x_ball, ~] = project_point_to_line(ball_w{i}, max_x, max_y);
    %     [proj_min_x_org, ~] = project_point_to_line(org_w{i}, min_x, min_y);
    %     [proj_max_x_org, ~] = project_point_to_line(org_w{i}, max_x, max_y);
    % %
    %     px_ball = proj_min_x_ball:0.01:proj_max_x_ball;
    %     px_org = proj_min_x_org:0.01:proj_max_x_org;
    py_bsvm = -(ball_w{i}(1) * px_ball + ball_w{i}(3))/ball_w{i}(2);
    py_org = -(org_w{i}(1) * px_org + org_w{i}(3))/org_w{i}(2);
    py_kmeans = -(kmeans_w{i}(1) * px_kmeans + kmeans_w{i}(3))/kmeans_w{i}(2);
    
    h_kmeans = plot(px_kmeans, py_kmeans, '--m', 'LineWidth', 3);
    hold on;
    h_org = plot(px_org, py_org, '-.g', 'LineWidth', 3);
    hold on;
    h_ball = plot(px_ball, py_bsvm, '-k', 'LineWidth', 3);
    if i == 1
        hleg =legend([h_org, h_kmeans, h_ball], 'Standard SVM', 'K-means SVM', 'Ball SVM');
        set(hleg,'FontSize', 14);
    end
    axis equal;
    %     org_dec_val{i} = test_features * org_w{i};
    %     org_ap(i) = myAP(org_dec_val{i}, test_labels',1);
end