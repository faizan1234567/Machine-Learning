function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
C = [ 0.01 0.03 0.1 0.3 1 3 10 30];
sigma =[ 0.01 0.03 0.1 0.3 1 3 10 30];
result = zeros(numel(sigma),numel(C));
x1 = [1 2 1]; x2 = [0 4 -1];
%l=X;
%gaussian = @(sig) gaussianKernel(X,l,sig);
for i = 1 :numel(C)
    for j = 1:numel(sigma)
        Temp_c = C(i);
        Temp_sig = sigma(j);
  model= svmTrain(X,y,Temp_c,@(x1,x2) gaussianKernel(x1,x2,Temp_sig));
  predictions = svmPredict(model,Xval);
  result(j,i) = mean(double(predictions ~= yval));
 % result((8*row+j),1)= mean(double(predictions ~= yval));
    end
end
% =========================================================================
v = indmin(result);
sigma = sigma(v(1));
C = C(v(2));

end
