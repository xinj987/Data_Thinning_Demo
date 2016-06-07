function [T, Likelihood] = ComputeLikelihood(node, x, obsIdx)
% Calculate the log likelihood
% Input
%   node - a leaf node
%   Xt - observation in mini-batches (p x N_t)
%   obsIdx - observed indices of each column of Xt (Boolean p x 1)
% Output
%   Likelihood - log likelihood of the x with respect to the mixture element described in the node
%   T - Likelihood plus the weight of current node

p = length(x);
r = length(node.spread);

q1 = log(node.weight);
q2 = -p/2*log(2*pi);
q3 = -1/2*(sum(log(node.spread)) + (p-r)*log(node.deviate));

mu = node.center(obsIdx);
V = node.basis(obsIdx,:);
invV = node.invBasis(:,obsIdx);
sigma = node.deviate;
w = invV * (x(obsIdx)-mu);  % Distance from vt to node center, in U axis
proj = V * w; % Projected distance of vt to node, in U basis
res = x(obsIdx) - mu - proj;  % residual distance
q4 = (w.*(1./node.spread))'*w + 1/sigma*(res'*res);  % log likelihood

Likelihood = q2+q3-1/2*q4;  % Don't take weight into account for likelihood under single Gaussian, or we bias patches towards the highest-weighted subspace
T = q1+Likelihood; % Likelihood with weight of this node

return;
