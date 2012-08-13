function ball = min_enclose_ball(data, label, opt)

C = opt.C_svdd;
eta_c = opt.eta_c;
eta_r = opt.eta_r;
max_iter_center = opt.max_iter_c;
max_iter_radius = opt.max_iter_r;
max_iter_global = opt.max_iter_svdd;
conv_thresh = opt.svdd_conv_thresh;

n = size(data,1);
sgd_batch = n;   % Using entire data for now

% Initialization
center_curr = mean(data,1);
radius_curr = 3 * mean(std(data,0,1));
center_best = center_curr;
center_prev = center_curr;
radius_best = radius_curr;
radius_prev = radius_curr;

iter_g = 1;
while 1
    % Phase 1: Update center
    iter_c = 1;
    obj_prev = compute_obj(data, C, center_best, radius_best);
    obj_best = obj_prev;
    while 1
        grad_c = 0;
        pm = randperm(n);
        for i = 1 : sgd_batch
            idx = pm(i);
            xi = data(idx,:);
            if norm(xi - center_curr) > radius_curr
                grad_c = grad_c + 2 * (center_curr - xi);
            end
        end
        center_curr = center_curr - eta_c * C * grad_c;
        obj_curr = compute_obj(data, C, center_curr, radius_best);
        obj_dec = obj_prev - obj_curr;
        % Update only if the objective is less than the current best
        if obj_curr < obj_best
            obj_best = obj_curr;
            center_best = center_curr;
        end
        fprintf('Phase 1 - Update Center: iter = %d, curr_obj = %f, obj_dec = %08f,  norm(center) = %f\n', iter_c, obj_curr, obj_dec, norm(center_curr));
        if iter_c > max_iter_center || (norm(center_curr - center_prev))/(norm(center_curr)+eps) < conv_thresh
            break;
        end
        iter_c = iter_c + 1;
        obj_prev = obj_curr;
        center_prev = center_curr;
    end
    
    % Phase 2: Update radius
    iter_r = 1;
    obj_prev = compute_obj(data, C, center_best, radius_best);
    obj_best = obj_prev;
    while 1
        grad_r = 2 * radius_curr;
        pm = randperm(n);
        for i = 1 : sgd_batch
            idx = pm(i);
            xi = data(idx,:);
            if norm(xi - center_curr) > radius_curr
                grad_r = grad_r - C * 2 * radius_curr;
            end
        end
        radius_curr = radius_curr - eta_r * grad_r;
        obj_curr = compute_obj(data, C, center_best, radius_curr);
        obj_dec = obj_prev - obj_curr;
        if obj_curr < obj_best
            obj_best = obj_curr;
            radius_best = radius_curr;
        end
        fprintf('Phase 2 - Update Center: iter = %d, curr_obj = %f, obj_dec = %08f,  norm(radius) = %f\n', iter_r, obj_curr, obj_dec, norm(radius_curr));
        if iter_r > max_iter_radius || (norm(radius_curr - radius_prev))/(norm(radius_curr)+eps) < conv_thresh
            break;
        end
        iter_r = iter_r + 1;
        obj_prev = obj_curr;
        radius_prev = radius_curr;
    end
    fprintf('Big Loop: Iter = %d, best_obj = %f, norm(best_center) = %f, norm(best_radius) = %f\n', iter_g, obj_best, norm(center_best), norm(radius_best));
    iter_g = iter_g + 1;
    if iter_g > max_iter_global || (iter_c == 1 && iter_r == 1)
        break;
    end
end
ball.center = center_best;
ball.radius = radius_best;
ball.numpts = n;
ball.label = label;

function obj = compute_obj(data, C, center, radius)
data = mat2cell(data, ones(size(data,1),1));
temp_sum = sum(cellfun(@(x) max([0, norm(x-center)^2 - radius^2]), data));
obj = radius^2 + C * temp_sum;