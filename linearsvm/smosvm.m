function smo = smosvm(sample, target)
global smo
% Initial struct  
smo = struct;
smo.b = 0;
smo.C = .01;
smo.tol = 0.001;
smo.epsilon = 0.001;
smo.alpha = zeros(size(target));
smo.Error = -target;

numChanged = 0;
examineAll = 1;
while numChanged > 0 || examineAll
    numChanged = 0;
    if examineAll % check all
        for i = 1:size(sample, 2)
            numChanged = numChanged + ...
                examineExample(i, sample, target);
        end
    else
        for i = 1:size(sample, 2) % check non-bound examples
            if ~isBound(smo.alpha(i), smo.C)
                numChanged = numChanged + ...
                    examineExample(i, sample, target);
            end
        end
    end

    if examineAll == 1
        examineAll = 0;
    elseif numChanged == 0
        examineAll = 1;
    end
end
end

function f = examineExample(i, sample, target)
global smo
y2 = target(i);
alpha2 = smo.alpha(i);
r2 = smo.Error(i) * y2;
% check if alpha2 violates KKT
if ((r2 < -smo.tol && alpha2 < smo.C) ||...
    (r2 > smo.tol && alpha2 > 0))
    if (sum(~isBound(smo.alpha, smo.C)) > 1)
        j = findMaxStep(smo, i);
        if takeStep(j, i, sample, target)
            f = 1;
            return
        end
    end
    index = randsample(size(sample,2), size(sample, 2));
    for ii = 1:size(sample, 2)
        rndIndex = index(ii);
        if ~isBound(smo.alpha(rndIndex), smo.C)
            if takeStep(rndIndex, i, sample, target)
                f = 1;
                return
            end
        end
    end
    index = randsample(size(sample, 2), size(sample,2));
    for ii = 1:size(sample, 2)
        rndIndex = index(ii);
        if takeStep(rndIndex, i, sample, target)
            f = 1;
            return
        end
    end
end
f = 0;
return
end

function g = takeStep(i1, i2, sample, target)
global smo
if i1 == i2
    g = 0;
    return
end

alpha1 = smo.alpha(i1);
alpha2 = smo.alpha(i2);
x1 = sample(:, i1);
y1 = target(i1);
x2 = sample(:, i2);
y2 = target(i2);
s = y1*y2;
if y1 ~= y2
    L = max(0, alpha2 - alpha1);
    H = min(smo.C, smo.C + alpha2 - alpha1);
else
    L = max(0, alpha2 + alpha1 - smo.C);
    H = min(smo.C, alpha2 + alpha1);
end
if L == H
    g = 0;
    return
end
k11 = kernelFunc(x1, x1);
k12 = kernelFunc(x1, x2);
k22 = kernelFunc(x2, x2);
eta = k11 + k22 - 2*k12;
if eta > 0
    a2 = alpha2 + y2 * (smo.Error(i1) - smo.Error(i2)) / eta;
    if a2 < L
        a2 = L;
    elseif a2 > H
        a2 = H;
    end
else 
    f1 = y1 * (smo.Error(i1) + smo.b) - alpha1 * k11 - s * alpha2 * k12;
    f2 = y2 * (smo.Error(i2) + smo.b) - s * alpha1 * k12 - alpha2 * k22;
    L1 = alpha1 + s * (alpha2 - L);
    H1 = alpha1 + s * (alpha2 - H);
    LObj = L1*f1 + L*f2 + .5*L1*L1*k11 + .5*L*L*k22 + s*L*L1*k12;
    HObj = H1*f1 + H*f2 + .5*H1*H1*k11 + .5*H*H*k22 + s*H*H1*k12;
    if (LObj < HObj - smo.epsilon)
        a2 = L;
    elseif (LObj > HObj + smo.epsilon)
        a2 = H;
    else
        a2 = alpha2;
    end
end

if abs(a2 - alpha2) < smo.epsilon * (a2 + alpha2 + smo.epsilon)
    g = 0;
    return
end

a1 = alpha1 + s * (alpha2 - a2);

b1 = smo.Error(i1) + ...
    y1 * (a1 - alpha1) * k11 + y2 * (a2 - alpha2) * k12 + smo.b;
b2 = smo.Error(i2) + ...
    y1 * (a1 - alpha1) * k12 + y2 * (a2 - alpha2) * k22 + smo.b;

oldB = smo.b;
if a1 > 0 && a1 < smo.C
    smo.b = b1;
elseif a2 > 0 && a2 < smo.C
    smo.b = b2;
else
    smo.b = (b1 + b2) * 0.5;
end
smo.alpha(i1) = a1;
smo.alpha(i2) = a2;
smo.Error = updateError(smo, sample, target);
g = 1;
return
end

function j = findMaxStep(smo, i)
E1 = smo.Error(i);
% search all non-bound errors
error = smo.Error(~isBound(smo.alpha, smo.C));
error(i) = 0; % exclude itself
if E1 > 0
    [~, j] = min(error);
else
    [~, j] = max(error);
end
end

function error = updateError(smo, sample, target)
error = zeros(size(target));
for i = 1:size(sample, 2)
    out = evalSvm(smo, sample(:, i), sample, target);
    error(i) = out - target(i);
end
end

function k = kernelFunc(x1, x2)
% simple kernel: dot product
k = x1' * x2;
end

function flag = equal(a, b)
% b must be a number and a must be a vector
flag = (a < (b + eps)) & (a > (b - eps));
end

function flags = isBound(a, C)
% check whether a is on bound, either 0 or C 
isZero = equal(a, 0);
isC = equal(a, C);
flags = isZero | isC;
end
% EOF
