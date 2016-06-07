function [V, R] = multiPETRELS(V, R, Xt_cd, obsIdx, alpha)
% Update subspace U in mini-batches
% Inputs:
%   U - previous computed subspace (p x r)
%   R - cumulative coefficient covariance matrix (r x r)
%   Xt_cd - centered observation (after subtract mean of Gaussian from 
%     X_t) in mini-batches (p x N_t)
%   obsIdx - observed indices of each column of Xt (Boolean p x 1)
%   alpha - history factor, higher value means to consider node history more 
%     strongly, scaled by the number of leafIdx's
% Outputs:
%   U - updated subspace
%   R - updated cumulative coefficient covariance matrix

% compute coefficients
if ~isequal(obsIdx, true(length(Xt_cd(:,1)), 1))
    U_omega = V(obsIdx, :);
    A = U_omega \ Xt_cd(obsIdx,:);
else
    A = V' * Xt_cd;
end
% update R
R = alpha * R + A*A';
% update U
V(obsIdx,:) = V(obsIdx,:) + (Xt_cd(obsIdx,:)-V(obsIdx,:)*A) * A' / (R);
% orthonormalize U, when N_t is larger than r, this direct
% orthonormalization is fast
V = V * (V'*V)^(-.5);

return;