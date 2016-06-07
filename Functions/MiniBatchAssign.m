function [Tr, updatedLeaf, Score] = MiniBatchAssign(Tr, leafIdx, Xt, obsIdx, method, alpha)
% Perform Multi Assignment
% Inputs:
%   Tr - Tree of nodes. Each node relates to some subset of the data.
%   leafIdx - indexes of the leaf nodes (parents of the shadow nodes)
%   Xt - observation in mini-batches (p x N_t)
%   obsIdx - observed indices of each column of Xt (Boolean p x 1)
%   method - petrels_batch or petrels_fo. petrels_batch is the default
%     method, which process data in mini-batches. if petrels_fo is chosen,
%     the data will be processed sequentially.
%   alpha - forgetting factor for history, higher value means to consider node history more strongly, scaled by the number of leafIdx's
% Outputs:
%   Tr - updated tree of nodes, for all of the data.
%   updatedLeaf - list of leaves that were updated.
%   Score - for each patch, the negative log likelihood that it will occur in any of the leaf nodes

% assign each data point to its nearest leaf node and all its ancestors
% also assign to the nearer virtual node
% calculate the likelihood for each data point

nNodes = length(Tr);
v = cell(nNodes, 1);

[~,N] = size(Xt);
Score = zeros(N,1);


for i = 1:N
    leafSize = length(leafIdx);
    ll = zeros(leafSize, 1); % log likelihood under each leaf node
    T = zeros(leafSize, 1); % weighted (with weight of each node) log likelihood
    
    j = 0;
    % compute log-likelihood of each observation under each leaf node
    for n = leafIdx
        j = j+1;
        [T(j), ll(j)] = ComputeLikelihood(Tr(n), Xt(:,i), obsIdx);
    end
    [~, maxIdx] = max(ll);
    T_max = max(T);
    T = T - T_max;
    % negative log likelihood under the GMM model
    Score(i,1) = -T_max - log(sum(exp(T)));
    
    % Figure out closest leaf	
    closestLeaf = leafIdx(maxIdx);

    leftVir = Tr(closestLeaf).left;
    rightVir = Tr(closestLeaf).right;
    
    % compute the log likelihood under the virtual children nodes
    [~, ll_left] = ComputeLikelihood(Tr(leftVir), Xt(:,i), obsIdx);
    [~, ll_right] = ComputeLikelihood(Tr(rightVir), Xt(:,i), obsIdx);

    % list the virtual child that has higher likelihood as needing updates
    if ll_left > ll_right
        updateThis = leftVir;
    else
        updateThis = rightVir;
    end
    
    % list the leaf node (with highest likelihood) and its ancestors as
    % needing updates
    while updateThis ~= 0
        v{updateThis} = [v{updateThis}, i];
        updateThis = Tr(updateThis).father;
    end

end

% update each node by the assigned data points
updatedLeaf = [];
for i = 1:nNodes
    if ~isempty(v{i})
        % update in mini-batches
        Tr(i) = MiniBatchUpdate(Tr(i), Xt(:,v{i}), obsIdx, method, alpha);
        if ismember(i, leafIdx)
            updatedLeaf = [updatedLeaf, i]; % record updated nodes
        end
    end
end

virIdx = [Tr(leafIdx).left, Tr(leafIdx).right];

% Update the weight of Tr(i), based on the number of data points used to update it.
% If no points are used to update it, use this to update the weight, too.
% Only  do this update for virtual nodes, then update all other weights as
% sums of their chidren's weights
for i = virIdx
	nDataPoints = numel(v{i});
	Tr(i).weight = alpha*Tr(i).weight + (1-alpha)*(nDataPoints / N);
end

factor = sum([Tr(virIdx).weight]);

for i = virIdx
    Tr(i).weight = Tr(i).weight / factor;
end

% Step through the queue, and update all the nodes that are not
% virtual leaves to have weight equal to sum of their childrens' weights.
nodeQueue = leafIdx;
readyToUpdate = zeros(nNodes,1);
readyToUpdate(virIdx) = 2;
readyToUpdate(leafIdx) = 2;
while ~isempty(nodeQueue)
	i = nodeQueue(1);
	nodeQueue = nodeQueue(2:end);
	left = Tr(i).left;
	right = Tr(i).right;
	Tr(i).weight = Tr(left).weight + Tr(right).weight;

	father = Tr(i).father;
	if (father ~= 0)
		readyToUpdate(father) = readyToUpdate(father) + 1;
		if (readyToUpdate(father) == 2)
			nodeQueue = [nodeQueue, father];
		end
	end
end

return;