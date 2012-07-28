clear all; close all;
%% gaussian A
numOfPos = 1500;
muA = [2,3];
sigmaA = [1 1.8; 1.8 4.2];
dataA = mvnrnd(muA, sigmaA, numOfPos);

%% gaussian B
numOfNeg = 1600;
muB = [4,2];
sigmaB = [3 1.6; 1.6 2];
dataB = mvnrnd(muB, sigmaB, numOfNeg);


%% dataset = A + B
samples = [dataA;dataB];
targets = [ones(numOfPos, 1); ones(numOfNeg, 1)*-1];

%% learning classifier
w = logisticReg(samples', targets');
