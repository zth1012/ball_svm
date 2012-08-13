function [w_ball, neg_balls, neg_clusters, w_kmeans] = run_toy_example(train_label, train_feat, opt)

K = opt.K;
pos_idx = train_label == 1;
neg_idx = train_label == -1;

pos_feat = train_feat(pos_idx,:);
neg_feat = train_feat(neg_idx,:);
neg_cl_label = litekmeans(neg_feat', K);
K = max(neg_cl_label);

% Compute negative cluster centers
parfor k = 1 : K
    idx = (neg_cl_label == k);
    neg_clusters(k).center = mean(neg_feat(idx,:), 1);
    neg_clusters(k).numpts = sum(idx);
end

% TODO: will pre-allocate struct/cell enhance speed performance?
% Fit negative data with minimum enclosing balls
parfor k = 1 : K
    cl_idx = (neg_cl_label == k);
    neg_balls(k) = min_enclose_ball(neg_feat(cl_idx,:), -1, opt);
end

% Convert 'pos_feat' to the ball format
parfor k = 1 : size(pos_feat,1)
    pos_balls(k).center = pos_feat(k,:);
    pos_balls(k).radius = 0;
    pos_balls(k).numpts = 1;
    pos_balls(k).label = 1;
end

train_balls = [pos_balls, neg_balls];

w_kmeans = kmeans_svm(pos_feat, neg_clusters, opt);
w_ball = ball_svm(train_balls, opt);


function obj = compute_ball_svm_obj(w, C_n, C_p, pos_feat, neg_balls)
obj = 0.5 * norm(w(1:end-1),2)^2;
temp_sum = 0;
for k = 1 : length(neg_balls)
    temp_sum = temp_sum + max([0, neg_balls(k).numpts * (1 + neg_balls(k).center * w + neg_balls(k).radius * norm(w(1:end-1)))]);
end
fprintf('Negative objective: %f\n', temp_sum);
obj = obj + C_n * temp_sum;
temp_sum = 0;
for p = 1 : size(pos_feat,1)
    temp_sum = temp_sum + max([0, 1 - pos_feat(p,:) * w]);
end
fprintf('Positive objective: %f\n', temp_sum);
obj = obj + C_p * temp_sum;