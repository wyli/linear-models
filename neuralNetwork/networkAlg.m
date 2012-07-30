function networkAlg(samples, groundtruth)

figure1 = figure;
rate = 0.5;
% adding bias term to samples
X = [ones(1, size(samples, 2)); samples];

% initial weights
d = [5 2 1]; % number of cells in each layer
L = size(d, 2); % number of layers
W{1} = rand(size(X,1), d(1));
for l = 2 : L
	W{l} = rand(d(l-1), d(l));
end

index = 0;
while index < 10000
% stochastic gradient descent
% select a random sample
n = randsample(size(X,2), 1);
x = X(:,n);
delta = cell(L, 1);
out   = cell(L, 1);

% calc out
for l = 1:L
	for i = 1:d(l)
		if l == 1
			out{l}(i,1) = softFunc(W{1}(:,i)' * x(:));
			continue;
		end
		out{l}(i,1) = softFunc(W{l}(:,i)' * out{l-1}(:));
	end
end

% backpropgration
for l = L:-1:1
	for i = 1:d(l)
		if l == L
			delta{l}(i)= W{l}(i,:) * diffSoftFunc(out{l}(1), groundtruth(n));
		else
			sigma = W{l+1}(i,:) * delta{l+1}(:);
			xout = out{l}(i,1);
			delta{l}(i) = (1-xout^2) * sigma;
		end
	end
end

for l = 1:L
	if l == 1
		W{l} = W{l} - rate * x(:) * delta{l};
	else
		W{l} = W{l} - rate * out{l-1} * delta{l};
	end
end
index = index + 1;
if mod(index, 10) == 0
plotAll(figure1, samples, groundtruth, W);
drawnow;
end
end
end

function theta = softFunc(s)
x = exp(s);
y = exp(-s);
theta = (x - y) ./ (x + y);
end

function error = diffSoftFunc(s, y)
theta = softFunc(s);
error = 2 * (theta - y) * (1 - theta^2);
end

function plotAll(figHandle, samples, groundtruth, weights)
axes1 = axes('parent', figHandle);
xlim(axes1, [-8, 12]);
ylim(axes1, [-8, 12]);
hold on;
plotArea(figHandle, weights);
hold on;
plot(samples(1, groundtruth == 1), samples(2, groundtruth==1), '+', ...
	'Color', [1 0 0], 'linewidth', 2);
hold on
plot(samples(1, groundtruth == -1), samples(2, groundtruth==-1), '+',...
	'Color', [0 0.5 0], 'linewidth', 2);
hold on;
end

function plotArea(figHandle, weights)
axes1 = axes('parent',figHandle);
[x1 x2] = meshgrid(-8:12, -8:12);
y = zeros(size(x1));
for i = 1:size(x1, 1)
	for j = 1:size(x1, 2)
		x = [x1(i,j);x2(i,j)];
		y(i,j) = evalNetwork(x, weights);
	end
end
contour(x1, x2, y);
hold on;
end

function y = evalNetwork(x, W)
x = [1;x];
out = cell(size(W));
for l = 1:length(W)
	for i = 1:size(W{l}, 2)
		if l == 1
			out{l}(i,1) = softFunc(W{1}(:,i)' * x(:));
			continue;
		end
		out{l}(i,1) = softFunc(W{l}(:,i)' * out{l-1}(:));
	end
end
if out{end}(:) > 0
	y = 1;
else
	y = 0;
end
end
