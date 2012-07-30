function networkAlg(samples, groundtruth)

figure1 = figure;
rate = .1;
% adding bias term to samples
X = [ones(1, size(samples, 2)); samples];

% initial weights
d = [3 3 1]; % number of cells in each layer
L = size(d, 2); % number of layers
W{1} = -1 + 2 * rand(size(X,1), d(1));
for l = 2 : L
	W{l} = rand(d(l-1), d(l));
end

index = 0;
while index < 10000000

% stochastic gradient descent
% select a random sample
n = randsample(size(X,2), 1);
x = X(:,n);

% init space for all out put and derivatives
delta = cell(L, 1);
out   = cell(L, 1);

% evaluate network output
out = evalNetwork(x, W);

% backpropgration
for l = L:-1:1
	for i = 1:d(l)
		if l == L
			delta{l}(i)= W{l}(i,:) * diffSoftFunc(out{L}(1), groundtruth(n));
		else
			sigma = W{l+1}(i,:) * delta{l+1}(:);
			xout = out{l}(i,1);
			delta{l}(i) = (1-xout^2) * sigma;
		end
	end
end

% update weights
for l = 1:L
	if l == 1
		W{l} = W{l} - rate * x(:) * delta{l};
	else
		W{l} = W{l} - rate * out{l-1} * delta{l};
	end
end

% draw a map of weights
if mod(index, 1000) == 0
plotAll(figure1, samples, groundtruth, W);
drawnow;
end
% loop index update
index = index + 1;
fprintf('%d\n', index);
end % end of while loop
end % end of function

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
plotArea(figHandle, weights);
hold on;
plot(samples(1, groundtruth == 1), samples(2, groundtruth==1), '+', ...
	'Color', [1 0 0], 'linewidth', 2);
hold on
plot(samples(1, groundtruth == -1), samples(2, groundtruth==-1), '+',...
	'Color', [0 0 1], 'linewidth', 2);
hold on;
end

function plotArea(figHandle, weights)

[x1 x2] = meshgrid(-8:0.5:12, -8:0.5:12);
y = zeros(size(x1));
for i = 1:size(x1, 1)
	for j = 1:size(x1, 2)
		x = [1; x1(i,j);x2(i,j)];
		[~, r] = evalNetwork(x, weights);
		if r > 0
			y(i,j) = 1;
		else
			y(i,j) =-1;
		end
	end
end
contourf(x1, x2, y, 2);
hold on;
end

function [out, y] = evalNetwork(x, W)
out = cell(size(W));
for l = 1:length(W)
	for i = 1:size(W{l}, 2)
		if l == 1
			out{l}(i,1) = softFunc( W{1}(:,i)' * x(:) );
		else
			out{l}(i,1) = softFunc( W{l}(:,i)' * out{l-1}(:) );
		end
	end
end
y = out{end}(:);
end
