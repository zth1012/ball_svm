function ball = min_enclose_ball(data, C, eta_c, eta_r, max_iter_center, max_iter_radius, global_max_iter)
n = size(data,1);
sgd_batch = n;   % Using entire data for now

% Initialization
center = mean(data,1);
radius = 3 * mean(std(data,0,1));
best_center = center;
best_radius = radius;

g_iter = 1;
while 1
    % Phase 1: Update center
    iter = 1;
    prev_obj = compute_obj(data, C, best_center, best_radius);
    best_obj = prev_obj;
    while 1
        grad_c = 0;
        pm = randperm(n);
        for i = 1 : sgd_batch
            idx = pm(i);
            xi = data(idx,:);
            if norm(xi - center) > radius
                grad_c = grad_c + 2 * (center - xi);
            end
        end
        center = center - eta_c * C * grad_c;
        curr_obj = compute_obj(data, C, center, best_radius);
        obj_dec = prev_obj - curr_obj;
        % Update only if the objective is less than the current best
        if curr_obj < best_obj
            best_obj = curr_obj;
            best_center = center;
        end
        fprintf('Phase 1 - Update Center: iter = %d, curr_obj = %f, obj_dec = %08f,  norm(center) = %f\n', iter, curr_obj, obj_dec, norm(center));
        prev_obj = curr_obj;
        iter = iter + 1;
        if iter > max_iter_center
            break;
        end
    end
    
    % Phase 2: Update radius
    iter = 1;
    prev_obj = compute_obj(data, C, best_center, best_radius);
    best_obj = prev_obj;
    while 1
        grad_r = 2 * radius;
        pm = randperm(n);
        for i = 1 : sgd_batch
            idx = pm(i);
            xi = data(idx,:);
            if norm(xi - center) > radius
                grad_r = grad_r - C * 2 * radius;
            end
        end
        radius = radius - eta_r * grad_r;
        curr_obj = compute_obj(data, C, best_center, radius);
        obj_dec = prev_obj - curr_obj;
        if curr_obj < best_obj
            best_obj = curr_obj;
            best_radius = radius;
        end
        fprintf('Phase 2 - Update Center: iter = %d, curr_obj = %f, obj_dec = %08f,  norm(radius) = %f\n', iter, curr_obj, obj_dec, norm(radius));
        prev_obj = curr_obj;
        iter = iter + 1;
        if iter > max_iter_radius
            break;
        end
    end
    fprintf('Big Loop: Iter = %d, best_obj = %f, norm(best_center) = %f, norm(best_radius) = %f\n', g_iter, best_obj, norm(best_center), norm(best_radius));
    g_iter = g_iter + 1;
    if g_iter > global_max_iter
        break;
    end
end
ball.center = best_center;
ball.radius = best_radius;
ball.numpts = n;

function obj = compute_obj(data, C, center, radius)

temp_sum = 0;
for i = 1 : size(data,1)
    temp_sum = temp_sum + max([0, norm(data(i,:) - center)^2 - radius^2]);
end

obj = radius^2 + C * temp_sum;