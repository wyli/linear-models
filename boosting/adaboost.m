clear all; close all;

fprintf('testing adaboost\n');
fprintf('generate dataset\n');

% Gaussian a
numOfA = 70;
muA = [7, 7];
sigmaA = [2 0;0  2];
dataA = mvnrnd(muA, sigmaA, numOfA);

% Gaussian b
numOfB = 100;
muB = [-3, -7];
sigmaB = [2 0; 0 2];
dataB = mvnrnd(muB, sigmaB, numOfB);

% Gaussian c
numOfC = 100;
muC = [0, 3];
sigmaC = [3 1.5; 1.5 2];
dataC = mvnrnd(muC, sigmaC, numOfC);

% Gaussian d
numOfD = 50;
muD = [4, -1];
sigmaD = [2 2.5; 2.5 5];
dataD = mvnrnd(muD, sigmaD, numOfD);

% Gaussian f
numOfF = 100;
muF = [-5, 6];
sigmaF = [3 1.5; 1.5 2];
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

% format data
%samples = [dataA; dataB; dataH; dataD; dataC]';
%targets = [ones(numOfA+numOfB+numOfD+numOfH, 1);...
    %-1*ones(numOfC, 1)];
samples = [dataA; dataB; dataH; dataD; dataF; dataC]';
targets = [ones(numOfA+numOfB+numOfD+numOfF+numOfH, 1);...
    -1*ones(numOfC, 1)];
clear dataA dataB dataC dataD dataE dataF dataG dataH;
clear numOfA numOfB numOfC numOfD numOfE numOfF numOfG numOfH; 
clear muA muB muC muD muE muF muG muH;
clear sigmaA sigmaB sigmaC sigmaD sigmaE sigmaF sigmaG sigmaH;



%% start boosting
% initialise weights
w = ones(size(samples, 2), 1)./size(samples, 2);
T = 150; 
beta = zeros(T, 1);
h = zeros(3, T);

X = [samples; ones(1, size(samples, 2))];
ind = 1;
% search weak learner
for t = 1:T
    ht = weaklearner(w, X, targets);
    output = (ht'*X)';
    output = sign(output);
    e_i = double(output ~= targets)';
    e_t = e_i*w;
    if e_t < 0.5
        h(:,ind) = ht;
        ind = ind+1;
    else
        continue;
    end
    %fprintf('%f\n', e_t);
    beta(ind) = 0.5 * log((1-e_t)/e_t);
    w = w.*exp(-beta(ind).*targets.*output);
    w = w./sum(w); % normalize weights
    total = sum(sign(X'*h*beta) ~= targets);
    fprintf('%f\n', total);
end
xrange = [-15 15];
yrange = [-15 15];
inc = 0.5;
[x, y] = meshgrid(xrange(1):inc:xrange(2), yrange(1):inc:yrange(2));
image_size = size(x);
xy = [x(:) y(:)];
allout = [xy ones(length(xy), 1)] * h * beta;
idx = sign(allout);
%decisionmap = reshape(idx, image_size);
decisionmap = reshape(allout, image_size);
figure;
imagesc(xrange, yrange, decisionmap);
hold on;
set(gca, 'ydir', 'normal');

%cmap = [1 0.8 0.8; 0.95 1 0.95];
%colormap(cmap);
%colormap(gray);

% plot data
plot(samples(1, targets==1), samples(2, targets==1), '+',...
    'Color', [1 0 0], 'linewidth', 1.8);
hold on;
plot(samples(1, targets==-1), samples(2, targets==-1), '+',...
    'Color', [0 0.5 0], 'linewidth', 1.8);
%figure;
a = -15:15;
for t = 1:T
    plot(a, (h(1,t).*a+h(3,t))./(-h(2,t)));hold on;
end
ylim([-15,15]);
xlim([-15,15]);

