function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C_range = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigma_range = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

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

min_error = 999999;
min_error_C = 0.01;
min_error_sigma = 0.01;

for C = C_range
  for sigma = sigma_range
    # train model with training data
    model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma), 1e-3, 20);

    # do prediction on cross validation data
    predictions = svmPredict(model, Xval);
    prediction_error = mean(double(predictions ~= yval))
    if(prediction_error < min_error) 
      min_error = prediction_error;
      min_error_C = C;
      min_error_sigma = sigma;
    end

end

C = min_error_C
sigma = min_error_sigma
% =========================================================================

end
