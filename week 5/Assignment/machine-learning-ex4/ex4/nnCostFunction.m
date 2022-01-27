function [J grad ] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% PERFORMING FEED FORWARD PROPAGATION.

a1 = [ones(m,1) X];
Z2 = a1*Theta1';  % size of a1 (m x (n+1))  / n = 400 ( excluding BAIS unit)
                  % size of Theta1 ( h x (n+1))  
                  % Z2 should be (m x h)
a2 = sigmoid(Z2);
a2 = [ ones(m,1) a2];  
                  % size of a2 ( m x (h+1))
Z3 = a2*Theta2';
                  % size of Theta2 (r x (h+1))
                  % size of Z3 (m x r)
a3 = sigmoid(Z3);
                  % size of a3 (mxr)
h = a3;

Y = zeros(m,num_labels);
for j = 1: num_labels
    Y(:,j) = (y==j);
end
                 % size of Y matrix (m x r)
% COMPUTING COST WITHOUT REGULARIZTION
J = (1/m) * -(trace( Y' * log(h)) + trace(( 1 -Y)'*log( 1 -h)));
% Computing cost with regularization.
theta1 = Theta1;
theta1(:,1) =0;
theta2 = Theta2;
theta2(:,1) =0;
Theta = [ theta1(:);theta2(:)];
reg = (lambda/(2*m)) * sum(Theta.^2);
J = J + reg;
%Implementing Backpropagation Algorithm
 

d3 = h - Y; % matrix of errors 
            %size of d3 (m xr)
d2 = (d3 * (Theta2(:,2:end))) .*sigmoidGradient(Z2);
            % Theta2(:,2:end) is ( r x h)
             % (mxr)x(rxh).*(mxh) = (mxh)



Delta2 = d3' * a2; % (rxm)*(mxh+1) 
Delta1 = d2' *a1;  % (hxm) *(mx n+1)

Theta1_grad =  (1/m) .* Delta1 + (lambda/m)*theta1;

Theta2_grad =   (1/m) .* Delta2 + (lambda/m)*theta2;







% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
