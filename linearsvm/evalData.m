close all; clear all;
figure1 = figure;
%samples = [0.9 .7 .5 .45; .54 .8 .83 .8];
%targets = [-1; -1; 1; -1];
load('data.mat');
samples = X';
targets = y*2 -1;
smo = smosvm(samples, targets);
[x1 x2] = meshgrid(0:0.01:1, 0:0.01:1);
y = zeros(size(x1));
for i = 1:size(x1, 1)
    for j = 1:size(x1, 1)
        r(i, j) = evalSvm(smo, [x1(i, j); x2(i, j)], samples, targets)> 0;
    end
end
axes1 = axes('parent', figure1);
xlim(axes1, [0, 1]);
ylim(axes1, [0.4, 1]);
hold on;
plot(samples(1, targets'==1), samples(2, targets' ==1), '+', ...
    'Color', [1 0 0], 'linewidth', 2);
hold on;
plot(samples(1, targets'==-1), samples(2, targets'==-1), '+', ...
    'Color', [0 0.75 0], 'linewidth', 2);
contour(x1, x2, r, 1);
