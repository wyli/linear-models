samples = [0.814723686393179,0.126986816293506,0.632359246225410;0.905791937075619,0.913375856139019,0.097540404999410];
targets = [1;1;-1];
plot(samples(1,1:2), samples(2,1:2), 'r+');
hold on;
plot(samples(1,3), samples(2, 3), 'g+');
weights = smosvm(samples, targets);
x1 = [0:0.1:1];
x2 = [0:0.1:1];
[x1 x2] = meshgrid(0:0.05:1, 0:0.05:1);
y = zeros(size(x1));
for i = 1:size(x1, 1)
	for j = 1:size(x1, 2)
		x = [x1(i,j);x2(i,j)];
		r = weights' * x;
		if r > 0
			y(i,j) = 1;
		else
			y(i,j) = -1;
		end
	end
end
%contour(x1, x2, y, 1, 'linewidth', 2.5, 'Color', [0 0 0.75]);