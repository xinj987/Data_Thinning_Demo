%-------------------------------------
% This is the demo file for running the Online Thinning algorithm with a
% parking lot surveillance video
% 
% 
% Created by Xin Jiang (chlorisjiang@gmail.com)
% June 04, 2016
%-------------------------------------
clc; clear; close all;

% Add code directory and VL_FEAT toolbox
addpath Functions/ % Online Thinning functions
addpath demo_videos/ % input video file
addpath vl_feat_0.9.19/toolbox % VL_FEAT toolbox
run vl_setup; % setup VL_FEAT


% The Online Thinning algorithm uses the first frame for training, and
% process the video frame-by-frame. On each frame, the SIFT features are
% computed on a grid placed 33 x 33 pixels apart (can be changed by setting
% up the parameters for SIFT in Online_Thinning_Video), and the algorithm
% processes all SIFT features from the same frame in a mini-batch. An
% anomalouseness score is assigned by the Online Thinning algorithm for
% each SIFT feature vector (which corresponds to a small patch in the 
% image). After the computation of the scores, we flag the top 5% of
% patches with the highest score with red color in the output video.

% The output video will be stored in 'output_demo.avi'. Note that because
% the first frame is used for initialization, the actual output video will
% be one frame shorter than the input video.

% inVideoName = 'original_demo.avi';
inVideoName = 'original_demo.avi';
outVideoName = 'demo_videos/output_demo.avi';
Online_Thinning_Video(inVideoName,outVideoName);
