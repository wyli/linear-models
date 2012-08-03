function w = smosvm(sample, target)
global smo
% Initial struct  
smo = struct;
smo.b = 0;
smo.C = 0.1;
smo.tol = 0.001;
smo.epsilon = 0.001;
smo.alpha = rand(size(target));
smo.Error = updateError(smo, sample, target);
numChanged = 0;
examineAll = 1;
while numChanged > 0 || examineAll
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
weights = zeros(size(sample, 1), 1);
for i = 1:size(target, 1)
    weights = weights + target(i) * smo.alpha(i) * sample(:, i);
end
w = weights;
end

function f = examineExample(i, sample, target)
global smo
f = 1;
y2 = target(i);
alph2 = smo.alpha(i);
r2 = smo.Error(i) * y2;
% check if alpha2 violates KKT
if ((r2 < -smo.tol && alph2 < smo.C) ||...
    (r2 > smo.tol && alph2 > 0))
    if (sum(~isBound(smo.alpha, smo.C)) > 1)
        j = findMaxStep(smo, i);
        if takeStep(j, i, sample, target)
            return
        end
    end
    for ii = 1:size(sample, 2)
        if isBound(smo.alpha(ii), smo.C)
            if takeStep(ii, i, sample, target)
                return
            end
        end
    end
    for ii = 1:size(sample, 2)
        if takeStep(ii, i, sample, target)
            return
        end
    end
end
f = 0;
end

function g = takeStep(i1, i2, sample, target)
global smo
g = 0;
if i1 == i2
    return
end

alph1 = smo.alpha(i1);
alph2 = smo.alpha(i2);
x1 = sample(i1);
y1 = target(i1);
x2 = sample(i2);
y2 = target(i2);
s = y1*y2;
if y1 ~= y2
    L = max(0, alph2 - alph1);
    H = min(smo.C, smo.C + alph2 - alph1);
else
    L = max(0, alph2 + alph1 - smo.C);
    H = min(smo.C, alph2 + alph1);
end
if L == H
    return
end
k11 = kernelFunc(x1, x1);
k12 = kernelFunc(x1, x2);
k22 = kernelFunc(x2, x2);
eta = k11 + k22 - 2*k12;
if eta > 0
    a2 = alph2 + y2*(smo.Error(i1) - smo.Error(i2)) / eta;
    if a2 < L
        a2 = L;
    elseif a2 > H
        a2 = H;
    end
else 
    LObj = evalObjFunc(smo, sample, target, i2, L);
    HObj = evalObjFunc(smo, sample, target, i2, H);
    if LObj < HObj - smo.epsilon
        a2 = L;
    elseif LObj > HObj + smo.epsilon
        a2 = H;
    else
        a2 = alph2;
    end
end

if abs(a2 - alph2) < smo.epsilon * (a2 + alph2 + smo.epsilon)
    return
end

a1 = alph1 + s * (alph2 - a2);
b1 = smo.Error(i1) + ...
    y1 * (a1 - alph1) * k11 + y2 * (a2 - alph2) * k12 + smo.b;
b2 = smo.Error(i2) + ...
    y1 * (a1 - alph1) * k12 + y2 * (a2 - alph2) * k22 + smo.b;

if a1 > 0 && a1 < smo.C
    smo.b = b1;
elseif a2 > 0 && a2 < smo.C
    smo.b = b2;
else
    smo.b = (b1 + b2) / 2;
end
smo.alph(i1) = a1;
smo.alph(i2) = a2;
smo.Error = updateError(smo, sample, target);
g = 1;
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

function k = kernelFunc(x1, x2)
% simple kernel: dot product
k = x1' * x2;
end

function obj = evalObjFunc(smo, sample, target, i, alphi)
% evaluate object function (1/2)*\sigma(yyaaK(xx)) + sum(a) with updated alpha
alph = smo.alph;
alph(i) = alphi;
Q = zeros(size(target, 1));
for m = 1:size(target, 1)
    for n = 1:size(target, 1)
        Q(m,n) = target(m) * target(n) *...
            kernelFunc(sample(:, m) * sample(:, n));
    end
end
obj = alph' * Q * alph * 0.5 + sum(alph);
end

function error = updateError(smo, sample, target)
% evaluate errors (first evaluate weights)
% TODO: vectorise...
weights = zeros(size(sample, 1), 1);
for i = 1:size(target, 1)
    weights = weights + target(i) * smo.alpha(i) * sample(:, i);
end
for i = 1:size(sample, 2)
    error(i, 1) = (weights' * sample(:,i) - smo.b) - target(i);
end
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
