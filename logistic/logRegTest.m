clear all; close all;
%% gaussian A
numOfPos = 500;
muA = 1 + 2*rand(2,1);
sigmaA = [1 1.8; 1.8 4.2];
dataA = mvnrnd(muA, sigmaA, numOfPos);

%% gaussian B
numOfNeg = 500;
muB = 3*rand(2,1);
sigmaB = [3 1.6; 1.6 2];
dataB = mvnrnd(muB, sigmaB, numOfNeg);


%% dataset = A + B
samples = [dataA;dataB];
targets = [ones(numOfPos, 1); ones(numOfNeg, 1)*-1];

%% learning classifier
w = logisticReg(samples', targets');
