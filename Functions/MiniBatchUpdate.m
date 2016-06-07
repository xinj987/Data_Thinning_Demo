function [newNode] = MiniBatchUpdate(node, Xt, obsIdx, method, alpha)
% Perform Mini-Batch Update
% Inputs:
%   node - current node
%   Xt - observation in mini-batches (p x N_t)
%   obsIdx - observed indices of each column of Xt (Boolean p x 1)
%   method - petrels_batch or petrels_fo. petrels_batch is the default
%     method, which process data in mini-batches. if petrels_fo is chosen,
%     the data will be processed sequentially. 
%   alpha - history factor, higher value means to consider node history more 
%     strongly, scaled by the number of leafIdx's
% Outputs:
%   newNode - the updated node
newNode = node;

r = size(node.basis, 2);
[p,N] = size(Xt);
x_mean = mean(Xt, 2);

mu_old = node.center; % Gaussian mean \mu of node
V_old = node.basis; % subsapce V of node
Lambda_old = node.spread; % \Lambda of node
sigma_old = node.deviate; % sigma^2 of node, also tracked here
cum_ll_old = node.error; % cumulative likelihood e of node
R_old = node.R; % cumulative coefficient covariance of node



% update mean of Gaussian
mu_new = alpha*mu_old + (1-alpha)*x_mean;

% calculate center-subtracted values for each patch individually
centerSub = Xt - repmat(mu_new, [1,N]);

% update V and R
switch method
    case 'petrels_fo'
        [V_new, R_new] = multiOPAST(V_old, R_old, centerSub, obsIdx, alpha);
	case 'petrels_batch'
        [V_new, R_new] = multiPETRELS(V_old, R_old, centerSub, obsIdx, alpha);
end

if (N == 1)
    if sum(isnan(V_new)) > 0
        V_new = zeros(size(V_new));
        V_new(1) = 1;
    end
end


% update Lambda, estimated sigma^2, and cumulative log
% likelihood (cum_ll)
w = V_new \ centerSub;
if r == 1
    w_sq = (w*w')/N;
else
    % sum up the column-wise dot products
    w_sq = sum(dot(w,w))/N;
end
res = centerSub - V_new*w;
res_sq = sum(dot(res,res))/N;

Lambda_new = alpha*Lambda_old + (1-alpha)*w_sq;
sigma_new = alpha*sigma_old + (1-alpha)*(res_sq)/(p-r);

shrinkage = max([1, sqrt(sum(w_sq./Lambda_new))]);
ll = res_sq/(p-r) + (1-1/shrinkage)^2*(w_sq);
cum_ll = alpha*cum_ll_old + (1-alpha)*ll;


% save updated node parameters
newNode.basis = V_new;
newNode.invBasis = pinv(V_new);
newNode.center = mu_new;
newNode.spread = Lambda_new;
newNode.deviate = sigma_new;
newNode.error = cum_ll;
newNode.R = R_new;

return;