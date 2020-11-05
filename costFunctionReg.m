function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
n = size(X,2);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% computing the cost function.
z = X * theta;
h = sigmoid(z);
Theta = theta(2:end);
J = ((1/m) * -( y' * log(h) + (1 -y)' * log( 1- h))) + (lambda/(2*m)) *sum( Theta.^2);

% computing the gradients
grad(1) = (1/m)*(X(:,1)'*(h -y));
for i = 2: n
    grad(i) = (1/m) *(X(:,i)'*(h -y)) +(lambda/m)*theta(i);
end


% =============================================================

end
