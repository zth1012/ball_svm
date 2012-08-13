function best_w = kmeans_svm(pos_feat, neg_clusters, opt)

C_n = opt.C_n;
C_p = opt.C_p;
eta_w = opt.eta_w;
max_iter_w = opt.max_iter_w;
K = length(neg_clusters);

% Initialize w
w = zeros(size(pos_feat,2), 1);
best_w = w;
iter = 1;
prev_obj = compute_kmeans_svm_obj(w, C_n, C_p, pos_feat, neg_clusters);
best_obj = prev_obj;
w_prev = w;
while 1
    grad = w;
    grad(end) = 0;
    % Update gradient using the negative clusters
    for k = 1 : K
        if neg_clusters(k).center * w > -1
            grad = grad + C_n * neg_clusters(k).numpts * neg_clusters(k).center';
        end
    end
    % Update gradient using the positive data
    for p = 1 : size(pos_feat, 1)
        if pos_feat(p,:) * w < 1
            grad = grad - C_p * pos_feat(p,:)';
        end
    end
    w = w - eta_w * grad;
    
    curr_obj = compute_kmeans_svm_obj(w, C_n, C_p, pos_feat, neg_clusters);
    obj_dec = prev_obj - curr_obj;
    if curr_obj < best_obj
        best_obj = curr_obj;
        best_w = w;
    end
    fprintf('Training K-means SVM: Iter = %d, curr_obj = %f, obj_dec = %08f, norm(w) = %f\n', iter, curr_obj, obj_dec, norm(w));
    prev_obj = curr_obj;
    iter = iter + 1;
    if iter > max_iter_w ||(norm( w-w_prev)/(norm(w)+eps)<10^(-2))
        break;
    end
    w_prev = w;
end

function obj = compute_kmeans_svm_obj(w, C_b, C_p, pos_feat, neg_clusters)

obj = 0.5 * norm(w,2)^2;
temp_sum = 0;
for k = 1 : length(neg_clusters)
    temp_sum = temp_sum + max([0, neg_clusters(k).numpts * (1 + neg_clusters(k).center * w)]);
end
obj = obj + C_b * temp_sum;
temp_sum = 0;
for p = 1 : size(pos_feat,1)
    temp_sum = temp_sum + max([0, 1 - pos_feat(p,:) * w]);
end
obj = obj + C_p * temp_sum;