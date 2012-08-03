function smosvm(sample, target)

% Initial struct  
smo = struct;
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
smo.Error(i) = (smo.W(:,i)' * x2 - smo.b(i)) - y2;
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
smo.Error(i1) = (smo.W(:,i1)' * x1 - smo.b(i)) - y1;
smo.Error(i2) = (smo.W(:,i2)' * x2 - smo.b(i)) - y2;
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
smo.b = smo.Error(i1) + y1*(a1 - alph1)*k11 + y2*(a2 - alph2)*k12 + smo.b;
% update weights?
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
end

function e = findCache(smo, i, sample, target)
end

function k = kernelFunc(x1, x2)
k = x1' * x2;
end

function obj = evalObjFunc(smo, sample, target, i, alph)
end
% todo
% equal
% 
