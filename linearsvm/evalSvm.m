function label = evalSvm(smo, p, samples, targets)
W = 0;
for i = 1:size(samples, 2);
    W = W + targets(i) * smo.alpha(i) * samples(:,i);
end
label = W' * p - smo.b;
