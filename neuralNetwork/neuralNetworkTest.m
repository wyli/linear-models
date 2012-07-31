clear all; close all;
% Gaussian a
numOfA = 250;
muA = [6, 6];
sigmaA = [4 0;0  8];
dataA = mvnrnd(muA, sigmaA, numOfA);

% Gaussian b
numOfB = 100;
muB = [1, -6];
sigmaB = [2 2.5; 2.5 5];
dataB = mvnrnd(muB, sigmaB, numOfB);

% Gaussian c
numOfC = 250;
muC = [-1, 1];
sigmaC = [3 1.5; 1.5 11];
dataC = mvnrnd(muC, sigmaC, numOfC);

% Gaussian d
numOfD = 100;
muD = [5, 1];
sigmaD = [2 2.5; 2.5 5];
dataD = mvnrnd(muD, sigmaD, numOfD);

%samples = [dataA; dataB; dataC; dataD]';
%targets = [ones(numOfA+numOfB, 1); -1*ones(numOfC+numOfD,1)];

samples = [dataA; dataB; dataC;]';
targets = [ones(numOfA+numOfB, 1); -1*ones(numOfC,1)];
%plot(samples(1, targets==1), samples(2, targets==1), '+',...
%	'Color', [1 0 0], 'linewidth', 1.8);
%hold on;
%plot(samples(1, targets==-1), samples(2, targets==-1), '+',...
%	'Color', [0 0.5 0], 'linewidth', 1.8);

% neural network algorithm
SGD(samples, targets);

