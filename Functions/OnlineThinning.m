function [Tr, leafIdx, Score] = OnlineThinning(Tr, leafIdx, Xt, obsIdx, method, eps, gamma, alpha)
% Perform Online Thinning algorithm
% Inputs:
%   Tr - Tree of nodes. Each node relates to some subset of the data.
%   leafIdx - indexes of the leaf nodes (parents of the shadow nodes)
%   Xt - input observations (p x N_t)
%   obsIdx - observed indices of each column of Xt (Boolean p x 1)
%   method - petrels_batch or petrels_fo. petrels_batch is the default
%     method, which process data in mini-batches. if petrels_fo is chosen,
%     the data will be processed sequentially.
%   eps - Error bound, scaled by the number of leafIdx's
%   gamma - amount virtual leaves must improve error by, to be used, scaled 
%     by the number of leafIdx's
%   alpha - forgetting factor for history, higher value means to consider node 
%     history more strongly, scaled by the number of leafIdx's
% Outputs:
%   Tr - updated tree of nodes, for all of the data.
%   leafIdx - updated indexes of the leaf nodes
%   Score - for each column of Xt, the weighted negative log likelihood 
%     that it is from any of the leaf nodes



% assign each data point to a leaf node and its ancestors
% update each node with the data assigned
% calculate the Score
[Tr, updatedLeaf, Score] = MiniBatchAssign(Tr, leafIdx, Xt, obsIdx, method, alpha);

% split the tree
splitLeaf = [];
newLeaf = [];

for n = updatedLeaf
    leafErr = Tr(n).error;
    leftVir = Tr(n).left;
    rightVir = Tr(n).right;
    % weighted average of cumulated likelihood at the vitual children level
    virErr = (Tr(leftVir).error * Tr(leftVir).weight + Tr(rightVir).error * Tr(rightVir).weight) ...
        / (Tr(leftVir).weight + Tr(rightVir).weight);
    
    if (leafErr > eps) && (leafErr > virErr + gamma)
        leafIdx = setdiff(leafIdx, n);
        leafIdx = [leafIdx, leftVir, rightVir];

		splitLeaf = [splitLeaf, n];
		newLeaf = [newLeaf, leftVir, rightVir];
        
        Tr(end+1) = Tr(leftVir);
        [maxLen, maxIdx] = max(Tr(leftVir).spread);
        majorAxis = Tr(leftVir).basis(:, maxIdx);
        Tr(end).center = Tr(leftVir).center + sqrt(maxLen)/2*majorAxis;
        Tr(end).spread(maxIdx) = maxLen/4;
        Tr(end).left = -inf;
        Tr(end).right = -inf;
        Tr(end).father = leftVir;
        Tr(end).weight = Tr(leftVir).weight/2;
        Tr(leftVir).left = length(Tr);
        
        Tr(end+1) = Tr(leftVir);
        Tr(end).center = Tr(leftVir).center - sqrt(maxLen)/2*majorAxis;
        Tr(end).spread(maxIdx) = maxLen/4;
        Tr(end).left = -inf;
        Tr(end).right = -inf;
        Tr(end).father = leftVir;
        Tr(end).weight = Tr(leftVir).weight/2;
        Tr(leftVir).right = length(Tr);
        
        
        
        Tr(end+1) = Tr(rightVir);
        [maxLen, maxIdx] = max(Tr(rightVir).spread);
        majorAxis = Tr(rightVir).basis(:, maxIdx);
        Tr(end).center = Tr(rightVir).center + sqrt(maxLen)/2*majorAxis;
        Tr(end).spread(maxIdx) = maxLen/4;
        Tr(end).left = -inf;
        Tr(end).right = -inf;
        Tr(end).father = rightVir;
        Tr(end).weight = Tr(rightVir).weight/2;
        Tr(rightVir).left = length(Tr);
        
        Tr(end+1) = Tr(rightVir);
        Tr(end).center = Tr(rightVir).center - sqrt(maxLen)/2*majorAxis;
        Tr(end).spread(maxIdx) = maxIdx/4;
        Tr(end).left = -inf;
        Tr(end).right = -inf;
        Tr(end).father = rightVir;
        Tr(end).weight = Tr(rightVir).weight/2;
        Tr(rightVir).right = length(Tr);
    end
end

% merge the tree
[~, nPatches] = size(Xt);
abandonedLeaf = []; % leaves that aren't used any more
for i = leafIdx
    % delete obsolete nodes (weights too small)
    if (Tr(i).weight < (1/nPatches) * alpha^3)
        abandonedLeaf = [abandonedLeaf, i];
    end
end

updatedLeaf = [updatedLeaf, abandonedLeaf];

mergedLeaf = [];
removedVirtLeaf = [];
newLeaf = [];

% merge the tree
while ~isempty(updatedLeaf)
    n = updatedLeaf(1);
    updatedLeaf = updatedLeaf(2:end);
    
    if n ~= 1
        myFather = Tr(n).father;
        fatherErr = Tr(myFather).error;
        
        if Tr(myFather).left == n
            sib = Tr(myFather).right;
        else
            sib = Tr(myFather).left;
        end
        
        if (ismember(n, leafIdx)) && (ismember(sib, leafIdx))
            % weighted average of accumulative likelihood of the leaf node
            % and its sibling
            leafErr = (Tr(n).error * Tr(n).weight + Tr(sib).error * Tr(sib).weight) ...
                / (Tr(n).weight + Tr(sib).weight);
            
            if ((leafErr < eps) && (leafErr > fatherErr - gamma)) || ...
                    (ismember(n, abandonedLeaf) && ismember(sib, abandonedLeaf))
                leafIdx = setdiff(leafIdx, [n,sib]);
                leafIdx = [leafIdx, myFather];
                
                %Tr(myFather).weight = Tr(n).weight + Tr(sib).weight;
                updatedLeaf = setdiff(updatedLeaf, sib);
                abandonedLeaf = setdiff(abandonedLeaf, [n sib]);
				mergedLeaf = [mergedLeaf, [n, sib]];
				newLeaf = [newLeaf, myFather];
                
                del1 = Tr(n).left;
                del2 = Tr(n).right;
                del3 = Tr(sib).left;
                del4 = Tr(sib).right;
                del = [del1, del2, del3, del4];
                
				removedVirtLeaf = [removedVirtLeaf, del];

                for i = setdiff(1:length(Tr), del)
                    if ismember(Tr(i).left, del)
                        Tr(i).left = -inf;
                    else
                        Tr(i).left = Tr(i).left - sum(Tr(i).left > del);
                    end
                    
                    if ismember(Tr(i).right, del)
                        Tr(i).right = -inf;
                    else
                        Tr(i).right = Tr(i).right - sum(Tr(i).right > del);
                    end
                    
                    Tr(i).father = Tr(i).father - sum(Tr(i).father > del);
                end
                
                for i = 1:length(leafIdx)
                    leafIdx(i) = leafIdx(i) - sum(leafIdx(i) > del);
                end
                
                for i = 1:length(updatedLeaf)
                    updatedLeaf(i) = updatedLeaf(i) - sum(updatedLeaf(i) > del);
                end
                 
                for i = 1:length(abandonedLeaf)
                    abandonedLeaf(i) = abandonedLeaf(i) - sum(abandonedLeaf(i) > del);
                end
                
                Tr(del) = [];
            end
        end
    end
end
