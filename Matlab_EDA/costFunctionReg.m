function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
J = 0;
grad = zeros(size(theta));


h = sigmoid(X * theta);
J = -y .* log(h) - (1-y).* log(1-h);
J = sum(J);
J = J ./ m;
a = sum((theta(2:end)).^2);
a = (lambda / (2*m)) * a;
J = J + a;

grad = ((h - y)' * X)';
grad = grad ./ m;
grad(2:end) = grad(2:end) + (lambda .* theta(2:end)./ m);


% =============================================================

end
