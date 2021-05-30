function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%


sizeTh1 = size(Theta1); % 25 * 401
sizeTh2 = size(Theta2); % 10 * 26

sizeX = size(X); % 5000 * 400


X = [ones(size(X,1),1),X];
initH = Theta1 * X';

hyp = sigmoid(initH);

Z = hyp';


Z = [ones(size(Z,1),1),Z];
initHLayer2 = Theta2 * Z';

hypLayer2 = sigmoid(initHLayer2);


[row_id,col_id] = max(hypLayer2);

p = col_id';




% =========================================================================


end
