function smosvm(sample, target)

% Initial struct  
smo = struct;
smo.b = 0;
smo.C = 0.1;
smo.tol = 0.001;
smo.numChanged = 0;
smo.examineAll = 1;
smo.Error = zeros(size(target));
smo.alpha = zeros(size(target));

while smo.numChanged > 0 || smo.examineAll
    smo.numChanged = 0;
    if smo.examineAll % check all
        for i = 1:size(sample, 2)
            smo.numChanged = smo.numChanged + ...
                examineExample(smo, i, sample, target);
        end
    else
        for i = 1:size(sample, 2) % check non-bound examples
            if ~equal(alpha(i), 0) && ~equal(alpha(i), smo.C)
                smo.numChanged = smo.numChanged + ...
                    examineExample(smo, i, sample, target);
            end
        end
    end

    if smo.examineAll == 1
        smo.examineAll = 0;
    elseif smo.numChanged == 0
        smo.examineAll = 1;
    end
end
end

function f = examineExample(smo, i, sample, target);
x2 = sample(:,i);
y2 = target(i);
alph2 = smo.alpha(i);
smo.Error(i) = findCachedError(i);
r2 = Error(i) * y2;
if ((r2 < -smo.tol && alph2 < smo.C) ||...
    (r2 > smo.tol && alph2 > 0))
    if (sum(~equal(alpha, 0) && ~equal(alpha, smo.C)) > 1)
        j = findMaxStep(smo, i);
        if takeStep(smo, j, i, sample, target)
            return 1;
        end
    end
    for ii = 1:size(sample, 2)
        if ~equal(alpha(ii), 0) && ~equal(alpha(ii), smo.C)
            if takeStep(smo, ii, i, sample, target)
                return 1;
            end
        end
    end
    for ii = 1:size(sample, 2)
        if takeStep(smo, ii, i, sample, target)
            return 1;
        end
    end
return 0;
end
end


function g = takeStep(smo, i1, i2, sample, target)
if i1 == i2
    return 0;
end

alph1 = smo.alpha(i1);
alph2 = smo.alpha(i2);
x1 = sample(i1);
y1 = target(i1);
x2 = sample(i2);
y2 = target(i2);
smo.Error(i1) = findCachedError(i1);
smo.Error(i2) = findCachedError(i2);
s = y1*y2;
if y1 ~= y2
    L = max(0, alph2 - alph1);
    H = min(smo.C, smo.C + alph2 - alph1);
else
    L = max(0, alph2 + alph1 - smo.C);
    H = min(smo.C, alph2 + alph1);
end
if L == H
    return 0;
end
k11 = kernelFunc(x1, x1);
k12 = kernelFunc(x1, x2);
k22 = kernelFunc(x2, x2);
eta = k11 + k22 - 2*k12
if eta > 0
    a2 = alph2 + y2*(smo.Error(i1) - smo.Error(i2) / eta;
    if a2 < L
        a2 = L;
    elseif a2 > H
        a2 = H;
    end
elseif
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

if abs(a2 - alph2) < smo.epsilon * (a2 + alph2 + epsilon)
    return 0;
end

a1 = alph1 + s * (alph2 - a2);
b1 = smo.Error(i1) + ...
    y1 * (a1 - alph1) * k11 + y2 * (a2 - alph2) * k12 + smo.b;
b2 = smo.Error(i2) + ...
    y1 * (a1 - alph1) * k12 + y2 * (a2 - alph2) * k22 + smo.b;

if a1 > 0 && a1 < C
    smo.b = b1;
elseif a2 > 0 && a2 < C
    smo.b = b2;
else
    smo.b = (b1 + b2) / 2;
end
% update weights?
smo.alph(i1) = a1;
smo.alph(i2) = a2;
updateError(smo, sample, target);
end


function j = findMaxStep(smo, i)
E1 = smo.Error(i);
if E1 > 0
    [~, j] = min(smo.Error);
else
    [~, j] = max(smo.Error);
end
end

function b = equal(a, b)
a = double(a);
b = doubale(b);
return (a < b + 0.001) && (a > b - 0.001);
end

function e = findCachedError(smo, i)
return smo.Error(i);
end

function k = kernelFunc(x1, x2)
k = x1' * x2;
end

function obj = evalObjFunc(smo, sample, target, i, alphi)
alph = smo.alph;
alph(i) = alphi;
Q = zeros(size(target, 1));
for m = 1:size(target, 1)
    for n = 1:size(target, 1)
        sumLagrange = target(m) * target(n)...
             * alph(m) * alph(n) * kernelFunc(sample(:, m) * sample(:, n));
    end
end
obj = sumLagrange * 0.5 + sum(alph);
end

function error = updateError(smo, sample, target)
weights = zeros(size(sample, 1), 1);
for i = 1:size(target, 1)
    weights = weights + y(i) * alph(i) * sample(i);
end
for i = 1:size(sample, 2)
    smo.Error(i) = weights' * sample(i) - smo.b;
end
end
% todo
% equal
% 
