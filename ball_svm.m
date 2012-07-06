function ball_w = ball_svm(train_label, train_feat, opt)

C_svdd = opt.C_svdd;

pos_idx = train_label == 1;
neg_idx = train_label == -1;

pos_feat = train_feat(pos_idx,:);
neg_feat = train_feat(neg_idx,:);

% model = svmtrain2(train_label(neg_idx), neg_feat, ['-s 5 -c ' num2str(C_svdd)]);
ball_w = model;

function [radius, center] = min_enclose_ball(data, C)
n = size(data,1);
sgd_batch = n;   % Using entire data for now
pm = randperm(n);
% Initialization
center = mean(data, 2);
radius = 3 * mean(std(data,0,1));
for i = 1 : sgd_batch
    
end

