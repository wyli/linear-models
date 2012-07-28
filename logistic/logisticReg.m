function W = logisticReg(samples, targets)
figure1 = figure;
% set rate
rate = 0.3;

% adding bias components
X = [ones(1, size(samples, 2)); samples];

% initial weights
W = rand(1, 3);
grad = [1 1 1];

while sum(grad.^2) > 0.0002

	% gradient descent
	WX = W * X;
	YWX = targets .* WX;
	YX = repmat(targets, 3, 1) .* X;
	allgrad = YX ./ repmat((1+exp(YWX)), 3, 1);
	grad = sum(allgrad') ./ (-size(X,2));

	%update weights
	W = W - rate * grad;

	plotAll(figure1, samples, targets, W);
	drawnow;
end

function plotAll(figHandler, samples, groundtruth, weights)
axes1 = axes('parent', figHandler);
xlim(axes1, [-4, 10]);
ylim(axes1, [-4, 10]);
hold on;
plot(samples(1, groundtruth == 1), samples(2,groundtruth==1), '+', 'Color', [1 0 0], 'linewidth', 2);
hold on;
plot(samples(1, groundtruth == -1), samples(2, groundtruth==-1), '+', 'Color', [0 0.5 0], 'linewidth', 2);
hold on;
t = -4:0.5:10;
plot(t, (-weights(1) - weights(2)*t)/weights(3), 'blue', 'linewidth', 2);
end
end

