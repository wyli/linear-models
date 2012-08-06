function out = evalSvm(smo, p, samples, targets)
s = 0;
for i = 1:size(samples, 2);
    if smo.alpha(i) > 0
        s = s + targets(i) * smo.alpha(i) * kernelFunc(samples(:,i), p);
    end
end
out = s - smo.b;
