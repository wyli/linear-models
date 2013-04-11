function [weights] = weaklearner(p, X, y)

    W = rand(1, 3) * 2 - 1;
    grad = [1 1 1];
    targets = y';

    while sum(grad.^2) > 1e-7
        WX = W * X;
        YWX = targets .* WX;
        YX = repmat(targets, 3, 1) .* X;
        allgrad = YX ./ repmat((1+exp(YWX)),  3, 1);
        grad = allgrad*p ./(-size(X, 2));

        W = W - grad';
    end
    weights = W';
end
