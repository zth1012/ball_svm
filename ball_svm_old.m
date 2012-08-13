function w_best = ball_svm_old(train_balls, opt)

% Initialization
w_curr = 0.01 * rand(length(train_balls(1).center), 1);
w_prev = w_curr;
w_best = w_curr;

obj_curr = compute_obj(w_curr, train_balls, opt);
obj_prev = obj_curr;
obj_best = obj_curr;

iter = 1;
while 1
    grad = w_curr;
    grad(end) = 0;
%     pos_idx = arrayfun(@(x) x.label == 1, train_balls);
%     neg_idx = ~pos_idx;
    
    % Update gradient using the positive balls
    pidx = arrayfun(@(x) x.label==1, train_balls);
    pidx = find(pidx == 1);
    for p = 1 : numel(pidx)
        curr_ball = train_balls(pidx(p));
        if curr_ball.center*w_curr - curr_ball.radius * norm(w_curr(1:end-1))<1
            grad(1:end-1) = grad(1:end-1) + opt.C_p * curr_ball.numpts * (-curr_ball.center(1:end-1)' + curr_ball.radius * w_curr(1:end-1)/(norm(w_curr(1:end-1))+eps));
            grad(end) = grad(end) - opt.C_p * curr_ball.numpts;
        end
    end
    
    % Update gradient using the negative balls
    nidx = arrayfun(@(x) x.label==-1, train_balls);
    nidx = find(nidx == 1);
    for n = 1 : numel(nidx)
        curr_ball = train_balls(nidx(n));
        if -curr_ball.center*w_curr - curr_ball.radius * norm(w_curr(1:end-1))<1
            grad(1:end-1) = grad(1:end-1) + opt.C_n * curr_ball.numpts * (curr_ball.center(1:end-1)' + curr_ball.radius * w_curr(1:end-1)/(norm(w_curr(1:end-1))+eps));
            grad(end) = grad(end) + opt.C_n * curr_ball.numpts;
        end
    end
    
    w_curr = w_curr - opt.eta_w * grad;
    obj_curr = compute_obj(w_curr, train_balls, opt);
    obj_decr = obj_prev - obj_curr;
    if obj_curr < obj_best
        obj_best = obj_curr;
        w_best = w_curr;
    end
    figure(12004); hold on;
    plot(iter, obj_curr, '.');
    fprintf('Training Ball SVM: Iter = %d, obj_best = %f, obj_curr = %f, obj_decr = %08f, norm(w) = %f, bias = %f\n', iter, obj_best, obj_curr, obj_decr, norm(w_curr(1:end-1)), w_curr(end));
    if iter > opt.max_iter_w || (norm(w_curr-w_prev)/(norm(w_curr)+eps)<opt.bsvm_conv_thresh)
        break;
    end
    obj_prev = obj_curr;
    w_prev = w_curr;
    iter = iter + 1;
end

function obj = compute_obj(w, train_balls, opt)
C_p = opt.C_p;
C_n = opt.C_n;

pos_idx = arrayfun(@(x) x.label == 1, train_balls);
neg_idx = ~pos_idx;

pos_obj = C_p * sum(arrayfun(@(x) max([0, x.numpts*(1 - x.label*(x.center*w + x.radius*norm(w(1:end-1))))]), train_balls(pos_idx)));
neg_obj = C_n * sum(arrayfun(@(x) max([0, x.numpts*(1 - x.label*(x.center*w + x.radius*norm(w(1:end-1))))]), train_balls(neg_idx)));

obj = 0.5 * norm(w(1:end-1),2)^2 + pos_obj + neg_obj;