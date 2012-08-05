close all; clear all;
samples = [0.8,0.1,0.6,0.8;0.9,0.6,0.09,0.4];
targets = [-1;-1;1;-1];
smo = smosvm(samples, targets);
[x1 x2] = meshgrid(0:0.2:1, 0:0.2:1);
y = zeros(size(x1));
for i = 1:size(x1, 1)
    for j = 1:size(x1, 1)
        r(i, j) = evalSvm(smo, [x1(i, j); x2(i, j)], samples, targets)> 0;
    end
end
contourf(x1, x2, r);
hold on;
plot(samples(1,1), samples(2,1), 'g+');
hold on;
plot(samples(1,2), samples(2,2), 'g+');
hold on;
plot(samples(1,3), samples(2,3), 'r+');
hold on;
plot(samples(1,4), samples(2,4), 'g+');
evalSvm(smo, samples(:,1), samples, targets)
