clear all; close all;
% Gaussian a
numOfA = 600;
muA = [2, 4];
sigmaA = [3 0;0  4];
dataA = mvnrnd(muA, sigmaA, numOfA);

% Gaussian b
numOfB = 1;
muB = [3, 0];
sigmaB = [2 2.5; 2.5 5];
dataB = mvnrnd(muB, sigmaB, numOfB);

% Gaussian c
numOfC = 600;
muC = [-1, -1];
sigmaC = [2 2.5; 2.5 5];
dataC = mvnrnd(muC, sigmaC, numOfC);

% Gaussian d
numOfD = 1;
muD = [5, 1];
sigmaD = [2 2.5; 2.5 5];
dataD = mvnrnd(muD, sigmaD, numOfD);

samples = [dataA; dataC;]';
targets = [ones(numOfA, 1); -1*ones(numOfC,1)];

%plot(samples(1, targets==1), samples(2, targets==1), '+',...
%	'Color', [1 0 0], 'linewidth', 1.8);
%hold on;
%plot(samples(1, targets==-1), samples(2, targets==-1), '+',...
%	'Color', [0 0.5 0], 'linewidth', 1.8);

% neural network algorithm
networkAlg(samples, targets);

