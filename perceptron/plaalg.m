function weights = plaalg(samples, groundtruth)
figure1 = figure;
set(figure1, 'position', [650, 650, 800, 600]);

% initial weights (random)
weights = -1 + 2*rand(2,1);
labelNow = calcLabels(weights, samples);
plotAll(figure1, samples, groundtruth, weights);

while any(groundtruth-labelNow)
	drawnow;

	% all classification errors
	errors = (groundtruth - labelNow) ~= 0;
	errorsamples = samples(errors,:);
	errorYn = groundtruth(errors);

	% choose random samples from errors
	rindex = randsample(size(errorsamples,1), 1);

	% update weights
	weights = weights + (errorsamples(rindex,: ).*errorYn(rindex))';
	labelNow = calcLabels(weights, samples);
	plotAll(figure1, samples, groundtruth, weights);
end
end

function labelNow = calcLabels( weights, samples )
sumNow = samples(:,1) * weights(1) + samples(:,2) * weights(2); 
labelNow = (sumNow > 0) .* 2 - 1;
end

function plotAll(figHandler,samples, groundtruth, weights)
axes1 = axes('Parent',figHandler);
xlim(axes1,[-1 1]);
ylim(axes1,[-1 1]);

hold on;
% plot positive samples
plot(samples(groundtruth == 1, 1), samples(groundtruth ==1, 2), '+','Color', [1 0 0],'linewidth',2);
hold on;
% plot negative samples
plot(samples(groundtruth == -1, 1), samples(groundtruth == -1, 2), '+','Color', [0 0.5 0], 'linewidth', 2);

hold on;
% plot current classification boundary
t = -2:0.5:2;
plot(weights(2)*t, -weights(1)*t,'blue', 'linewidth', 2);
end
