function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));


% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

pred = X * theta;
Theta = theta;
Theta(1) =0;
J = (1/(2*m))* sum(( pred - y).^2) + ((lambda)/(2*m))*sum( Theta.^2);

%implementing gradient
grad(1) = (1/m)*X(:,1)'*(pred -y);
for i = 2 : size(X,2)
grad(i) = (1/m)*X(:,i)'*(pred -y) + (lambda/m)*theta(i);
end









% =========================================================================

grad = grad(:);

end
