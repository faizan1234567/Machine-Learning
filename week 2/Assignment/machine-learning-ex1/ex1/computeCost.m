function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
%J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.
prediction = X*theta;
% X is 97x2 matrix and Theta is 2x1 vector
%prediction should be 97x1 vector and y is also a 97x1 vector.
SqrError = (prediction - y).^2; % computing square error
SumSqrError = sum(SqrError); % and adding up all the square errors 
J =  SumSqrError/(2*m);     % finally the calculating the cost.




% =========================================================================

end
