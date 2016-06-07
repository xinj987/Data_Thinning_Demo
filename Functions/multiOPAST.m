function [V, R] = multiOPAST(V, R, Xt_cd, obsIdx, alpha)
% Update subspace U sequentially
% Inputs:
%   U - previous computed subspace (p x r)
%   R - cumulative coefficient covariance matrix (r x r)
%   Xt_cd - centered observation (after subtract mean of Gaussian from 
%     X_t) in mini-batches (p x N_t)%   obsIdx - observed indices of each column of Xt (Boolean p x 1)
%   alpha - history factor, higher value means to consider node history more 
%     strongly, scaled by the number of leafIdx's
% Outputs:
%   U - updated subspace
%   R - updated cumulative coefficient covariance matrix
[~,N] = size(Xt_cd);
lambda = ones(N,1);
lambda(1) = alpha;

for i = 1:N
    if ~isequal(obsIdx(:), true(length(Xt_cd(:,i)), 1))
        U_omega = V(obsIdx(:), :);
        y = U_omega \ Xt_cd(obsIdx(:), i);
    else
        y = V' * Xt_cd(:,i);
    end
    
    q = R * y;
    gamma = 1 / (1+y'*q);
    p = gamma * (Xt_cd(:,i)-V*y);
    p(~obsIdx(:)) = 0;
    % R = 1/lambda(i)*R-gamma*(q*q')
    % Computationally, the following equivalent expression works better for small alpha:
    R = 1/lambda(i)*R*gamma;

    norm_q = norm(q)^2;
    norm_p = norm(p)^2;
    
    tau = 1/norm_q * (1/sqrt(1+norm_p*norm_q)-1);
    p_prime = tau*V*q + (1+tau*norm_q)*p;
    V = V + p_prime*q';
end

return;