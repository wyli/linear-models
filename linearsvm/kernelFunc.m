function k = kernelFunc(x1, x2)
% simple kernel: dot product
% k = x1' * x2;

% radia
k = exp(-.05 * (x1-x2)'*(x1-x2));
end
