function SGD(samples, groundtruth)

figure1 = figure;
rate = .001;
% adding bias term to samples
X = [ones(1, size(samples, 2)); samples];

% initial weights
d = [15 15 1]; % number of cells in each layer
L = size(d, 2); % number of layers
%  input layer
W{1} = -0.5 + rand(size(X,1), d(1)-1);
% hidden layers
for l = 2 : L-1
	W{l} =-0.5 + rand(d(l-1), d(l)-1);
end
% output layer
W{L} = -0.5 + rand(d(L-1), 1);

index = 1;
errornum = 50
while errornum > 5

% stochastic gradient descent
% select a random sample
%n = mod(index, size(samples,2)) +1;
n = randsample(size(samples, 2), 1);
x = X(:,n);

% init space for all out put and derivatives
delta = cell(L+1, 1);
out   = cell(L+1, 1);

% evaluate network output
out = evalNetwork(x, W);
% backpropgration
delta{end} = diffSoftFunc(out{L+1}(1), groundtruth(n));
for l = L:-1:1
	for i = 1:size(out{l},1)
		if l == L
			sigma =W{L}(i,:)*delta{L+1}(1);
		else
			sigma = W{l}(i,:)*delta{l+1}(2:end);
		end
		delta{l}(i,1) = (1-out{l}(i)^2) * sigma;
	end
end

% update weights
for l = 1:L-1
	W{l} = W{l} - rate * out{l} * delta{l+1}(2:end)';
end
W{L} = W{L} - rate * out{L} * delta{L+1};


% loop index update
index = index + 1;
%fprintf('%d\n', index);

result = zeros(1, size(samples,2));
for t = 1:size(X, 2)
    [~, valu] = evalNetwork(X(:,t), W);
    result(t) = valu;
end
result = result > 0;
result = result*2 - 1;
errorCurr = sum((result'-groundtruth)~=0);
fprintf('%d %d\n', index, errorCurr);

% draw a map of weights
%if errornum < 100
if errorCurr < errornum
    plotAll(figure1, samples, groundtruth, W);
    drawnow;
	errornum = errorCurr;
end

end % end of while loop
end % end of function

function theta = softFunc(s)
	theta = (exp(s)- exp(-s)) ./ ( exp(s)+ exp(-s));
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
plot(samples(1, groundtruth == 1), samples(2, groundtruth==1), '+', ...
	'Color', [1 0 0], 'linewidth', 2);
hold on
plot(samples(1, groundtruth == -1), samples(2, groundtruth==-1), '+',...
	'Color', [0 0.75 0], 'linewidth', 2);
hold on;

[x1 x2] = meshgrid(-8:0.5:12, -8:0.5:12);
y = zeros(size(x1));
for i = 1:size(x1, 1)
	for j = 1:size(x1, 2)
		x = [1; x1(i,j);x2(i,j)];
		[~, r] = evalNetwork(x, weights);
		if r > 0
			y(i,j) = r;
		else
			y(i,j) =r;
		end
	end
end
contour(x1, x2, y, 1, 'linewidth', 2.5, 'Color', [0 0 0.75]);
end


function [out, y] = evalNetwork(x, W)
out = cell(size(W, 2)+1, 1); % output of layers
out{1,1} = x(:); % the input is used as first output
for l = 2:size(out, 1) % layers loop
	out{l}(1,1)  = 1;
	for i = 1:size(W{l-1}, 2) % nodes loop

		if l == size(out,1);  % final layer

			out{l}(1,1)   = softFunc( out{l-1, 1}' * W{l-1}(:,i) );
		else  % hidden layers

			out{l}(i+1,1) = softFunc( out{l-1, 1}' * W{l-1}(:,i) );
		end
	end
end
y = out{end}(:);
end
