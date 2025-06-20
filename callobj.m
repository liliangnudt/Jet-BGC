function [obj] = callobj(numview, X, Up, V, A, Z, alpha, eta, lambda, gamma)

D_m = diag(sum(Z,1));

term2 = -2 * trace(A' * V * Z) + eta * trace(A' * A * Z' * Z) + lambda * trace(A' * A * D_m) + gamma * trace(Z' * Z);
% term2 = -2 * trace(A' * V * Z) + eta * trace(A * Z' * Z * A') + lambda * trace(A * D_m * A') + gamma * trace(Z' * Z);

term1 = 0;
for p = 1 : numview
    temp1 = (X{p}-Up{p} * V)';
    temp2 = sqrt(sum(temp1.*temp1, 2));
    term1 = term1 + alpha(p)^2 * sum(temp2);
end

obj = term1 + term2;
end