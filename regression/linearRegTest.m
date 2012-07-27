function linearRegTest
%% generate data samples
w = -1 + 2*rand(1,2);
x = -1:0.1:1;
x = [ones(21,1), x'];
y = w*x';
y = y + rand(1,21);

w = linearReg(x, y');
plot(x(:,2), y, 'o', 'linewidth', 2.0);
hold on;
plot(x(:,2), w*x', '-r', 'linewidth', 2.0);

function w = linearReg(x, y)
%% samples in x are row vectors.
%% y is a column vector of targets.

pseudoInverse = inv(x'*x) * x';
w = pseudoInverse * y;
w = w';
end
end
