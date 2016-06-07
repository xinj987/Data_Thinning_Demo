function [node] = batchIni(X, r)
% Initializes a node with all the data in X.
% Inputs:
%   X - initialization data.  matrix of size p x N
%   r - Dimension of the subspace
% Outputs:
%   node - a node, which corresponds to all of the different patches.

[p,N] = size(X);
c = mean(X, 2);
centerSub = X - repmat(c, [1,N]);  % X - mean of its patch

try
    [V,S,~] = svd(centerSub, 'econ');
catch exception
    % Add Gaussian noise if ill-posed. Experimentally, it works pretty well.
    noise = zeros(p,N);
    variance = var(X');
    for i = 1:p
        noise(i,:) = 0.1 * normrnd(0, variance(i), 1,N);
    end
    centerSub = centerSub + noise;
    [V,S,~] = svd(centerSub, 'econ');
end


node.basis = V(:, 1:r);
node.invBasis = pinv(node.basis);
node.center = c;
% node.centerWeight = c'*c;
if (N > 1)
    S = diag(S).^2 / N;
    node.spread = S(1:r);
    node.deviate = sum(S(r+1:end)) / (p-r);
else
    % Special case - there is only one element here, so technically there is no
    % spread or deviation.  Give these a small non-zero value so that there isn't
    % a divide-by-zero error.
    % If they're too small, the probability distributions are very poorly conditioned.
    S = S.^2;
    node.spread = 1e-08;
    node.deviate = 1e-12;
end

beta = node.basis' * centerSub;
shrinkage = sqrt(diag(beta' * diag(1./node.spread) * beta));
shrinkage(shrinkage < 1) = 1;
node.error = sum(S(r+1:end)) + mean((1-1/shrinkage).^2.*sum(beta.^2, 1));

node.father = 0;
node.left = -inf;
node.right = -inf;
node.weight = 0;
node.R = eye(r);
end
