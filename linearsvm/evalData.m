close all; clear all;
figure1 = figure;
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

samples = [dataA; dataB; dataH; dataD; dataF; dataC;dataG]';
targets = [ones(numOfA+numOfB+numOfD+numOfF+numOfH, 1);...
             -1*ones(numOfC+numOfG,1)];
clear dataA dataB dataH dataD dataF dataC dataG;
%samples = [0.9 .7 .5 .45; .54 .8 .83 .8];
%targets = [-1; -1; 1; -1];
%load('data3.mat');
%index = randsample(size(y, 1), 200, 1);
%X = X';
%samples = X(:,index);
%targets = y(index,1)*2 -1;
% index = randsample(size(targets, 1), 450, 1);
% %X = X';
% samples = samples(:,index);
% targets = targets(index,1);
% training svm
tic;
smo = smosvm(samples, targets);
timeForSVM = toc;

% evaluate svm on meshgrid in order to visualise decision boundary
[x1 x2] = meshgrid(-15:0.2:15, -15:0.2:15);
y = zeros(size(x1));
for i = 1:size(x1, 1)
    for j = 1:size(x2, 1)
        r(i, j) = evalSvm(smo, [x1(i, j); x2(i, j)], samples, targets) > 0;
    end
end
axes1 = axes('parent', figure1);
xlim(axes1, [-10, 15]);
ylim(axes1, [-15, 15]);
% plot samples
plot(samples(1, targets'==1), samples(2, targets' ==1), '+', ...
    'Color', [1 0 0], 'linewidth', 2);
hold on;
plot(samples(1, targets'==-1), samples(2, targets'==-1), '+', ...
    'Color', [0 0.75 0], 'linewidth', 2);
% plot decision boundary
contour(x1, x2, r, 1, 'linewidth', 2, 'Color', [0 0 0.75]);
hold on;
% plot support vectors
sv = samples(:, smo.alpha>0);
for i = 1:size(sv, 2)
    plot(sv(1,i), sv(2,i), 's', 'linewidth', 0.5,'markersize', 12, 'Color', [0.2, 0.2, 0.2]);
end
timeForSVM
