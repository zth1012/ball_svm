function [best_w, neg_balls, kmeans_w] = ball_svm(train_label, train_feat, opt)

eta_c = opt.eta_c;
eta_r = opt.eta_r;
max_iter_c = opt.max_iter_c;
max_iter_r = opt.max_iter_r;
global_max_iter = opt.global_max_iter;
C_svdd = opt.C_svdd;
K = opt.K;
C_n = opt.C_n;
C_p = opt.C_p;
eta_w = opt.eta_w;
max_iter_w = opt.max_iter_w;

pos_idx = train_label == 1;
neg_idx = train_label == -1;

pos_feat = train_feat(pos_idx,:);
neg_feat = train_feat(neg_idx,:);
cl_label = litekmeans(neg_feat', K);
K = max(cl_label);

% Compute cluster centers
for k = 1 : K
    idx = (cl_label == k);
    neg_clusters{k}.center = mean(neg_feat(idx,:), 1);
    neg_clusters{k}.numpts = sum(idx);
end

kmeans_w = kmeans_svm(pos_feat, neg_clusters, opt);

% Fit data with minimum enclosing balls
for k = 1 : K
    cl_idx = (cl_label == k);
    neg_balls{k} = min_enclose_ball(neg_feat(cl_idx,:), C_svdd, eta_c, eta_r, max_iter_c, max_iter_r, global_max_iter);
%     neg_balls{k}.radius = 0;
end

% Visualization for 2D data
if size(train_feat, 2) == 3
    figure,
    scatter(neg_feat(:,1),neg_feat(:,2),50,'^');
    hold on;
    scatter(pos_feat(:,1),pos_feat(:,2),50,'rs'); 
    for k = 1 : K
        plot(neg_clusters{k}.center(1), neg_clusters{k}.center(2), 'x', 'LineWidth', 3, 'MarkerSize', 20, 'MarkerEdgeColor', 'm');
        hold on;
        circle(neg_balls{k}.center(1:2), neg_balls{k}.radius, 100, '-k');
    end
end

% Initialize w
w = 0.1 * rand(size(train_feat,2), 1);
best_w = w;
best_iter = 1;
iter = 1;
prev_obj = compute_ball_svm_obj(w, C_n, C_p, pos_feat, neg_balls);
best_obj = prev_obj;
figure(12004); clf;
w_prev = w;
while 1
    grad = w;
    grad(end) = 0;
    % Update gradient using the negative balls
    for k = 1 : K
        if neg_balls{k}.center * w + neg_balls{k}.radius * norm(w(1:end-1)) > -1
%             fprintf('Negative violation: %f\n', neg_balls{k}.center * w + neg_balls{k}.radius * norm(w));
            grad(1:end-1) = grad(1:end-1) + C_n * neg_balls{k}.numpts * (neg_balls{k}.center(1:end-1)' + neg_balls{k}.radius * w(1:end-1)/(norm(w(1:end-1)) + eps));
            grad(end) = grad(end) + C_n * neg_balls{k}.numpts;
        end
    end
    % Update gradient using the positive data
    for p = 1 : size(pos_feat, 1)
        if pos_feat(p,:) * w < 1
            grad = grad - C_p * pos_feat(p,:)';
        end
    end
    w = w - eta_w * grad;
%     
%     [px_proj_ball, ~] = arrayfun(@(x,y) project_point_to_line(w, x, y), train_feat(:,1), train_feat(:,2));
%     px_ball = min(px_proj_ball):0.01:max(px_proj_ball);
%     py_bsvm = -(w(1) * px_ball + w(3))/w(2);
%     h_ball = plot(px_ball, py_bsvm, '-m', 'LineWidth', 5);
    
    curr_obj = compute_ball_svm_obj(w, C_n, C_p, pos_feat, neg_balls);
    obj_dec = prev_obj - curr_obj;
    if curr_obj < best_obj
        best_obj = curr_obj;
        best_w = w;
        best_iter = iter;
    end
    figure(12004);hold on;
    plot(iter, curr_obj,'.', 'MarkerSize', 5);
    fprintf('Training Ball SVM: Iter = %d, curr_obj = %f, obj_dec = %08f, norm(w) = %f\n', iter, curr_obj, obj_dec, norm(w));
    fprintf('Training Ball SVM: best_iter = %d, best_obj = %f, norm(best_w) = %f\n', best_iter, best_obj, norm(best_w));
    prev_obj = curr_obj;
    iter = iter + 1;
    if iter > max_iter_w ||(norm( w-w_prev)/(norm(w)+eps)<10^(-3))
        break;
    end
    
    w_prev = w;
end
figure(1);

function obj = compute_ball_svm_obj(w, C_n, C_p, pos_feat, neg_balls)
obj = 0.5 * norm(w(1:end-1),2)^2;
temp_sum = 0;
for k = 1 : length(neg_balls)
    temp_sum = temp_sum + max([0, neg_balls{k}.numpts * (1 + neg_balls{k}.center * w + neg_balls{k}.radius * norm(w(1:end-1)))]);
end
fprintf('Negative objective: %f\n', temp_sum);
obj = obj + C_n * temp_sum;
temp_sum = 0;
for p = 1 : size(pos_feat,1)
    temp_sum = temp_sum + max([0, 1 - pos_feat(p,:) * w]);
end
fprintf('Positive objective: %f\n', temp_sum);
obj = obj + C_p * temp_sum;