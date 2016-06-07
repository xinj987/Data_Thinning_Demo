function [Tr, leafIdx] = iniTree(X, r, eps0)
% Initialize tree of nodes
% Inputs:
%   X - initialization data.  matrix of size [p x N]
%   r - subspace demension
%   eps0 = Error bound - if error for current nodes is greater than this, add a new level of nodes. 
% Outputs:
%   Tr - a tree of nodes, for all of the data.
%   leafIdx - indexes of the leaf nodes (parents of the virtual children nodes)


[~,N] = size(X);

if N < 5*r
    error('Not enough observations for initialization. Please provide at least %d initialization points.',5*r);
    return;
end

Tr = batchIni(X, r);  % Initialize the head node, with data from all patches.
Tr.weight = 1;

leafIdx = 1;
inspected = 1;
dataThisNode = {1:N};

while ~isempty(inspected)
    eps = Tr(inspected(1)).error;
   
    % if error for current tree is above some threshhold	
    if eps > eps0
		% split patches into 2 clusters
        [dataIdx, ~] = kmeans(X(:, dataThisNode{inspected(1)})', 2);
        
        leftIdx = dataThisNode{inspected(1)}(dataIdx == 1);
        rightIdx = dataThisNode{inspected(1)}(dataIdx == 2);
       
        % Add two new children to the tree.
        % NOTE: until recently, there was a requirement that there must be
        % at least 5*r elements in each data cluster here.  There may still
        % be bugs for the special case where this is not true.
		if (numel(leftIdx) >= r && numel(rightIdx) >= r)
			Tr(end+1) = batchIni(X(:, leftIdx), r);
			Tr(end).weight = length(leftIdx)/N;

			Tr(end+1) = batchIni(X(:, rightIdx), r);
			Tr(end).weight = length(rightIdx)/N;

			Tr(inspected(1)).left = length(Tr)-1;
			Tr(inspected(1)).right = length(Tr);
			Tr(end-1).father = inspected(1);
			Tr(end).father = inspected(1);

			inspected = [inspected, length(Tr)-1, length(Tr)];
			dataThisNode = [dataThisNode, leftIdx, rightIdx];
			leafIdx(leafIdx == inspected(1)) = [];
			leafIdx = [leafIdx, length(Tr)-1, length(Tr)];
		end

    end
    
    inspected(1) = [];
end

% For each item still in leafIdx (leaf nodes, which were not given children because error
% was below the threshhold), add a left and right shadow node.  they have quick and dirty
% aggregate approximations of the appropriate statistics.
for i = leafIdx
    
    % kmeans division to be added
    
    Tr(end+1) = Tr(i);
    [maxLen, maxIdx] = max(Tr(i).spread);
    majorAxis = Tr(i).basis(:, maxIdx);
    Tr(end).center = Tr(i).center + sqrt(maxLen)/2*majorAxis;
%     Tr(end).centerWeight = Tr(end).centerWeight' * Tr(end).centerWeight;
    Tr(end).spread(maxIdx) = maxLen/4;
    Tr(end).left = -inf;
    Tr(end).right = -inf;
    Tr(end).father = i;
    Tr(end).weight = Tr(i).weight/2;
    Tr(i).left = length(Tr);
    
    Tr(end+1) = Tr(i);
    Tr(end).center = Tr(i).center - sqrt(maxLen)/2*majorAxis;
%     Tr(end).centerWeight = Tr(end).centerWeight' * Tr(end).centerWeight;
    Tr(end).spread(maxIdx) = maxLen/4;
    Tr(end).left = -inf;
    Tr(end).right = -inf;
    Tr(end).father = i;
    Tr(end).weight = Tr(i).weight/2;
    Tr(i).right = length(Tr);
end
