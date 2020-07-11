function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

H = X * Theta';
# dot product R to remove r=0 items
error = (H - Y) .* R;
error_square = error.^2;

# note R must be logical to do the operation: https://www.mathworks.com/help/matlab/ref/logical.html
# For example:
# A = magic(5); I = logical(eye(5)); A(I)
# J = sum(error_square(R)) /2;
J = sum(sum(error_square)) /2;

regularization_J = (sum(sum(Theta.^2)) + sum(sum(X.^2))) * lambda /2;
J = J + regularization_J;

X_grad = (error * Theta);
regularization_X_grad = lambda * X;
X_grad = X_grad + regularization_X_grad;

Theta_grad = ((error)' * X);
regularization_Theta_grad = lambda * Theta;
Theta_grad = Theta_grad + regularization_Theta_grad;

% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
