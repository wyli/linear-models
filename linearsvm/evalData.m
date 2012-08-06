close all; clear all;
figure1 = figure;
%samples = [0.9 .7 .5 .45; .54 .8 .83 .8];
%targets = [-1; -1; 1; -1];
load('data3.mat');
index = randsample(size(y, 1), 200, 1);
X = X';
samples = X(:,index);
targets = y(index,1)*2 -1;
tic;
smo = smosvm(samples, targets);
timeForSVM = toc;
[x1 x2] = meshgrid(-0.8:0.005:0.8, -0.8:0.005:0.8);
y = zeros(size(x1));
for i = 1:size(x1, 1)
    for j = 1:size(x2, 1)
        r(i, j) = evalSvm(smo, [x1(i, j); x2(i, j)], samples, targets) > 0;
    end
end
axes1 = axes('parent', figure1);
xlim(axes1, [-0.8, 0.4]);
ylim(axes1, [-0.8, 0.8]);
hold on;
plot(samples(1, targets'==1), samples(2, targets' ==1), '+', ...
    'Color', [1 0 0], 'linewidth', 2);
hold on;
plot(samples(1, targets'==-1), samples(2, targets'==-1), '+', ...
    'Color', [0 0.75 0], 'linewidth', 2);
contour(x1, x2, r, 1, 'linewidth', 2, 'Color', [0 0 0.75]);
hold on;
% plot support vectors
sv = samples(:, smo.alpha>0);
for i = 1:size(sv, 2)
    plot(sv(1,i), sv(2,i), 's', 'linewidth', 1);
end
timeForSVM
