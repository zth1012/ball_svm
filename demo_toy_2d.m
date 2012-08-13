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
C_p = [30];   % Insert more values for parameter tuning
C_n = [10];
C_svdd = [0.1];
% K = [1 2 5 10 20];
K = 16;

cnt = 0;
for i = 1 : numel(C_p)
    for j = 1 : numel(C_n)
        for k = 1 : numel(K)
            for m = 1 : numel(C_svdd)
                cnt = cnt + 1;
                opts{cnt}.C_p = C_p(i);
                opts{cnt}.C_n = C_n(j);
                opts{cnt}.C_svdd = C_svdd(m);
                opts{cnt}.eta_c = 0.001;
                opts{cnt}.eta_r = 0.001;
                opts{cnt}.eta_w = 0.000001;
                opts{cnt}.max_iter_c = 500;
                opts{cnt}.max_iter_r = 500;
                opts{cnt}.max_iter_w = 1000;
                opts{cnt}.bsvm_conv_thresh = 0.001;
                opts{cnt}.max_iter_svdd = 50;
                opts{cnt}.svdd_conv_thresh = 0.01;
                opts{cnt}.K = K(k);
            end
        end
    end
end

org_w = cell(cnt,1);
w_ball = cell(cnt,1);
neg_balls = cell(cnt,1);

for i = 1 : cnt
    [w_ball{i}, neg_balls{i}, neg_clusters{i}, w_kmeans{i}] = run_toy_example(train_labels, train_features, opts{i});
    model = train(double(train_labels), sparse(train_features), ['-s 0 -c ' num2str(opts{i}.C_p) ' -q 1']);
    org_w{i} = model.w';
    
    [px_proj_org, ~] = arrayfun(@(x,y) project_point_to_line(org_w{i}, x, y), train_features(:,1), train_features(:,2));
    [px_proj_ball, ~] = arrayfun(@(x,y) project_point_to_line(w_ball{i}, x, y), train_features(:,1), train_features(:,2));
    [px_proj_kmeans, ~] = arrayfun(@(x,y) project_point_to_line(w_kmeans{i}, x, y), train_features(:,1), train_features(:,2));
    px_org = min(px_proj_org):0.01:max(px_proj_org);
    px_ball = min(px_proj_ball):0.01:max(px_proj_ball);
    px_kmeans = min(px_proj_kmeans):0.01:max(px_proj_kmeans);
    
    py_bsvm = -(w_ball{i}(1) * px_ball + w_ball{i}(3))/w_ball{i}(2);
    py_org = -(org_w{i}(1) * px_org + org_w{i}(3))/org_w{i}(2);
    py_kmeans = -(w_kmeans{i}(1) * px_kmeans + w_kmeans{i}(3))/w_kmeans{i}(2);
    
    figure, hold on;
    scatter(neg_feat(:,1),neg_feat(:,2),50,'^');
    hold on;
    scatter(pos_feat(:,1),pos_feat(:,2),50,'rs');
    for k = 1 : length(neg_balls{i})
        plot(neg_clusters{i}(k).center(1), neg_clusters{i}(k).center(2), 'x', 'LineWidth', 3, 'MarkerSize', 20, 'MarkerEdgeColor', 'm');
        hold on;
        circle(neg_balls{i}(k).center(1:2), neg_balls{i}(k).radius, 100, '-g');
    end
    
    h_kmeans = plot(px_kmeans, py_kmeans, '--m', 'LineWidth', 5);
    hold on;
    h_org = plot(px_org, py_org, '-.k', 'LineWidth', 5);
    hold on;
    h_ball = plot(px_ball, py_bsvm, '-g', 'LineWidth', 5);
    if i == 1
        hleg =legend([h_org, h_kmeans, h_ball], 'Standard SVM', 'K-means SVM', 'Ball SVM');
        set(hleg,'FontSize', 16, 'FontWeight', 'bold', 'Location', 'NorthWest');
    end
    set(gca,'FontSize',16)

    %     org_dec_val{i} = test_features * org_w{i};
    %     org_ap(i) = myAP(org_dec_val{i}, test_labels',1);
end