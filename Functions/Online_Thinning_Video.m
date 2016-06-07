%-------------------------------------
% This is the demo file for running the Online Thinning algorithm with a
% parking lot surveillance video
% 
% Created by Xin Jiang (chlorisjiang@gmail.com)
% June 04, 2016
% 
% Input: 
% inVideoName - input video name
% outVideoName - output video name
% 
% Output: None (result written into output video)
%-------------------------------------
function Online_Thinning_Video(inVideoName,outVideoName)

%-------------------------------------
% Read in video Data
%-------------------------------------
video_data = VideoReader(inVideoName);
nRows = get(video_data, 'Height');
nCols = get(video_data, 'Width');
TotalFrames = get(video_data, 'NumberOfFrames');
nFrames = TotalFrames-1; % the first frame will be used for training

%-------------------------------------
% Set up parameters
%-------------------------------------
% tuning parameters for SIFT
% These control the size of the area each SIFT feature is computed upon
BinSize = 3;
Mag = 2;
WinSize = 2.6;

% tuning for Online Thinning
eps_init = 2; % error level for initialization
eps = 1.5; % error level for 
gamma = 0.2;
alpha = 0.7; % forgetting factor
p = 128; % length of SIFT vectors
r = 1; % subspace dimension, for small image patches, r=1 is good enough. Use larger r for more complex scenes.
obsInx = true(p,1); % observed index

%-------------------------------------
% Initialize Online Thinning tree structure
%-------------------------------------
% compute the number of patches (n1 by n2)
n1 = round(nCols/(BinSize*Mag*WinSize)*0.5-2);
n2 = round(nRows/(BinSize*Mag*WinSize)*0.5-2);

% construct grid on which the SIFT features are computed
FrameGrid = [kron(round(linspace(1+BinSize*Mag*WinSize*2,nCols-2*BinSize*Mag*WinSize,n1)),ones(1,n2));...
     repmat(round(linspace(1+2*BinSize*Mag*WinSize,nRows-2*BinSize*Mag*WinSize,n2)),1,n1);...
     BinSize*ones(1,n1*n2);...
     zeros(1,n1*n2)];

% reload video data
video_data = VideoReader(inVideoName);
fprintf('Initialization: ');
img = readFrame(video_data);
img = im2single(rgb2gray(img));

% compute SIFT features
[Grid, Feat] = vl_sift(img,'frames',FrameGrid,'orientations',...
                'EdgeThresh',1000,'PeakThresh', 0,...
                'Magnif',Mag,'WindowSize',WinSize,...
                'NormThresh',0);
Features = double(Feat);

% compute actual patch locations
vari = zeros(1,size(Grid,2));
for k = 1:size(Grid,2)
    x1 = round(max(1,Grid(1,k)-Grid(3,k)*Mag*WinSize));
    x2 = round(min(nCols,Grid(1,k)+Grid(3,k)*Mag*WinSize));
    y1 = round(max(1,Grid(2,k)-Grid(3,k)*Mag*WinSize));
    y2 = round(min(nRows,Grid(2,k)+Grid(3,k)*Mag*WinSize));
    patch = double(img(y1:y2,x1:x2));
    vari(k) = (max(patch(:))-min(patch(:))).^(1/4);
end
% normalize feature vectors over brightness of current frame
Features =  Features./max(Features(:)).*repmat(vari,128,1)./max(vari);

% initialize tree structure
[Tr, leafIdx] = iniTree(Features, r, eps_init);
fprintf('done.\n');

%% 
%-------------------------------------
% Online Thinning for frame >=2
%-------------------------------------
Score = cell(nFrames,1); % save all anomalouseness score
Tr_save = cell(nFrames,1); % save all tree structures
AllGrid = cell(nFrames,1); % save all patch locations
AllFeatures = cell(nFrames,1); % save all feature vectores

fprintf('Processing frame: ');
for t = 1:nFrames
    fprintf('%03d/%03d',t+1,nFrames+1);
    img = readFrame(video_data);
	img = im2single(rgb2gray(img));
    
    % compute SIFT features
    [Grid, Feat] = vl_sift(img,'frames',FrameGrid,'orientations',...
                    'EdgeThresh',1000,'PeakThresh', 0,...
                    'Magnif',Mag,'WindowSize',WinSize,...
                    'NormThresh',0);
    Features = double(Feat);
    
    % find actual patch locations
    vari = zeros(1,size(Grid,2));
    for k = 1:size(Grid,2)
        x1 = round(max(1,Grid(1,k)-Grid(3,k)*Mag*WinSize));
        x2 = round(min(nCols,Grid(1,k)+Grid(3,k)*Mag*WinSize));
        y1 = round(max(1,Grid(2,k)-Grid(3,k)*Mag*WinSize));
        y2 = round(min(nRows,Grid(2,k)+Grid(3,k)*Mag*WinSize));
        patch = double(img(y1:y2,x1:x2));
        vari(k) = max(patch(:))-min(patch(:));
    end
    % normalize feature vectors over brightness of current frame
    Features =  Features./max(Features(:)).*repmat(vari,128,1)./max(vari);
    
    % save patch locations, feature vectors, and tree structure
    AllGrid{t} = Grid;
    AllFeatures{t} = Features;
    [Tr, leafIdx, Score{t}] = OnlineThinning(Tr, leafIdx, AllFeatures{t}, ...
        obsInx, 'petrels_batch', eps, gamma, alpha);
    Tr_save{t} = Tr;
    if t<nFrames
        fprintf('\b\b\b\b\b\b\b');
    end
end
fprintf('\nScore computation complete. Now creating video...\n');

%% 
%-------------------------------------
% Visualization
%-------------------------------------
% find top 5% patches with highest anomalousness scores
% compute the threshold
thresh = zeros(nFrames,1);
for t= 1:nFrames
    thresh(t) = prctile(Score{t}, 95);
end

% flag top 5% patches with red color and save video
outVideo = VideoWriter(outVideoName);
outVideo.FrameRate = 10;
open(outVideo);
video_data = VideoReader(inVideoName);
for t= 1:nFrames
    img = readFrame(video_data);
    % display original image in gray scale only
    img_g = rgb2gray(img);
    img(:,:,1) = img_g;
    img(:,:,2) = img_g;
    img(:,:,3) = img_g;
    Grid = AllGrid{t};
    Score_t = Score{t}; % scores of current frame
    
    % color top 5% patches with highest scores with red
    for k = 1:length(Score_t)
        % if not top 5%, skip
        if Score_t(k) <= thresh(t)
            continue; 
        end
        % find actual patch locations
        x1 = round(max(1,Grid(1,k)-Grid(3,k)*Mag*WinSize-1));
        x2 = round(min(nCols,Grid(1,k)+Grid(3,k)*Mag*WinSize+1));
        y1 = round(max(1,Grid(2,k)-Grid(3,k)*Mag*WinSize-1));
        y2 = round(min(nRows,Grid(2,k)+Grid(3,k)*Mag*WinSize+1));
        % make patch red
        img(y1:y2,x1:x2,1) = uint8(1.6*img_g(y1:y2,x1:x2));  % R
        img(y1:y2,x1:x2,2) = uint8(0.7*img_g(y1:y2,x1:x2));  % G
        img(y1:y2,x1:x2,3) = uint8(0.7*img_g(y1:y2,x1:x2));  % B
    end
    % crop out edges where we did not compute the scores
    [d1,d2,~] = size(img);
    EdgeSize = round(BinSize*Mag*WinSize);
    img_crop = img(EdgeSize:d1-EdgeSize, EdgeSize:d2-EdgeSize, :);
    % write current frame to outVideo
    writeVideo(outVideo, img_crop);
end
close(outVideo)
fprintf('Result stored in %s.\n',outVideoName);
return;