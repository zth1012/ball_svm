% The last entry of 'w' is assumed to be the bias term
function w_best = ball_svm(train_balls, opt)

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
    
    % Update gradient using the positive balls
    % 1. Find positive balls violating the constraints
    vio_pos = arrayfun(@(x) x.label==1 && x.center*w_curr - x.radius * norm(w_curr(1:end-1))<1, train_balls);
    if sum(vio_pos)>0
        pos_grads = cell2mat(arrayfun(@(x) x.numpts * (-x.center(1:end-1)' + x.radius * w_curr(1:end-1)/(norm(w_curr(1:end-1))+eps)), train_balls(vio_pos), 'UniformOutput', false));
        grad(1:end-1) = grad(1:end-1) + opt.C_p * sum(pos_grads,2);
        grad(end) = grad(end) - opt.C_p * sum(arrayfun(@(x) x.numpts, train_balls(vio_pos)));
    end
    
    % Update gradient using the negative balls
    % 1. Find negative balls violating the constraints
    vio_neg = arrayfun(@(x) x.label==-1 && -x.center*w_curr - x.radius * norm(w_curr(1:end-1))<1, train_balls);
    if sum(vio_neg)>0
        neg_grads = cell2mat(arrayfun(@(x) x.numpts * (x.center(1:end-1)' + x.radius * w_curr(1:end-1)/(norm(w_curr(1:end-1))+eps)), train_balls(vio_neg), 'UniformOutput', false));
        grad(1:end-1) = grad(1:end-1) + opt.C_n * sum(neg_grads,2);
        grad(end) = grad(end) + opt.C_n * sum(arrayfun(@(x) x.numpts, train_balls(vio_neg)));
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