clear all; close all;
% Gaussian a
numOfA = 70;
muA = [7, 7];
sigmaA = [4 0;0  8];
dataA = mvnrnd(muA, sigmaA, numOfA);

% Gaussian b
numOfB = 100;
muB = [-3, -7];
sigmaB = [2 2.5; 2.5 5];
dataB = mvnrnd(muB, sigmaB, numOfB);

% Gaussian c
numOfC = 100;
muC = [0, 3];
sigmaC = [4 1.5; 1.5 5];
dataC = mvnrnd(muC, sigmaC, numOfC);

% Gaussian d
numOfD = 50;
muD = [4, -1];
sigmaD = [2 2.5; 2.5 5];
dataD = mvnrnd(muD, sigmaD, numOfD);

% Gaussian f
numOfF = 50;
muF = [-5, 6];
sigmaF = [3 1.5; 1.5 8];
dataF = mvnrnd(muF, sigmaF, numOfF);

% Gaussian G
numOfG = 50;
muG = [5, -6];
sigmaG = [3 1.5; 1.5 8];
dataG = mvnrnd(muG, sigmaG, numOfG);

% Gaussian H
numOfH = 50;
muH = [10, -2];
sigmaH = [3 1.5; 1.5 8];
dataH = mvnrnd(muH, sigmaH, numOfH);

%samples = [dataA; dataB; dataC; dataD]';
%targets = [ones(numOfA+numOfB, 1); -1*ones(numOfC+numOfD,1)];

samples = [dataA; dataB; dataH; dataD; dataF; dataC;dataG]';
targets = [ones(numOfA+numOfB+numOfD+numOfF+numOfH, 1); -1*ones(numOfC+numOfG,1)];
%plot(samples(1, targets==1), samples(2, targets==1), '+',...
%	'Color', [1 0 0], 'linewidth', 1.8);
%hold on;
%plot(samples(1, targets==-1), samples(2, targets==-1), '+',...
%	'Color', [0 0.5 0], 'linewidth', 1.8);

% neural network algorithm
SGD(samples, targets);

