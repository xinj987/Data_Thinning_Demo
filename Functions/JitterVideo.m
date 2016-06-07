%-------------------------------------
% This is the demo file for running the Online Thinning algorithm with an
% artificially jittered parking lot surveillance video
%
% This function adds artificial rotation to the input video file and store
% it in a new video file
% 
% Created by Xin Jiang (chlorisjiang@gmail.com)
% June 06, 2016
% 
% Input: 
% inVideoName - input video name
% outVideoName - output video name
% 
% Output: None (result written into output video)
%-------------------------------------
function JitterVideo(inVideoName,outVideoName)

video_data = VideoReader(inVideoName);
nRows = get(video_data, 'Height');
nCols = get(video_data, 'Width');
TotalFrames = get(video_data, 'NumberOfFrames');

%% tansform image
fprintf('Adding jitter to the video...');
outVideo = VideoWriter(outVideoName);
outVideo.FrameRate = 10;
open(outVideo);
video_data = VideoReader(inVideoName);
for t =1:TotalFrames
    % read in frame
    img = readFrame(video_data);
    img = im2single(rgb2gray(img));

    if t > 1
        theta_old = theta;
    end
	% randomly generate rotation intensity
    theta = max(min(randn(),0.5),-0.5)/pi/10;
    if t > 1
        theta = min(max(theta, theta_old-pi/10),theta_old+pi/10);
    end
    % generate affine transform matrix
    A = [cos(theta) sin(theta) 0; -sin(theta) cos(theta) 0; 0 0 1];
    affTrans = affine2d(A);
    
    % warp image
    img_aff = imwarp(img,affTrans);    
    [ymax,xmax] = size(img_aff);
    
    % pad image and save to video
    padSize = 5;
    padImg = padarray(img_aff,[padSize,padSize]);
    shifty = ceil((ymax+2*padSize-nRows)/2)+round(rand()*2*padSize-padSize);
    shiftx = ceil((xmax+2*padSize-nCols)/2)+round(rand()*2*padSize-padSize);
    writeVideo(outVideo, padImg(shifty+1:nRows+shifty,shiftx+1:nCols+shiftx));
end
close(outVideo);
fprintf('complete.\n');
fprintf('Jittered video stored in %s.\n',outVideoName);
return;