function k = kernelFunc(x1, x2)
% simple kernel: dot product
% k = x1' * x2;

% radia
k = exp(-.01 * (x1-x2)'*(x1-x2));
end
