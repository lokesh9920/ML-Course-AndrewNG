function [J grad] = nnCostFunction(nn_params, ...
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

sizeTheta1 = size(Theta1); %25*401
sizeTheta2 = size(Theta2); % 10*26
    
% Setup some useful variables
m = size(X, 1);


sizeX = size(X); %5000*400
sizeY = size(y); % 5000*1 
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
a1 = X;
X = [ones(size(X,1),1),X]; %appending theta0 i.e 1 to X
% new X is 5000*401 size

thetaXTrans = Theta1 * X';
Xforlayer3 = thetaXTrans';
Xforlayer3BeforeSigmoid = Xforlayer3;
Xforlayer3 = sigmoid(Xforlayer3);

%IMPORTANT
%append ones only after calculating sigmoid. Vice-Versa is WRONG
a2 = [ones(size(Xforlayer3,1),1),Xforlayer3]; %5000*26


thetaa2Trans = Theta2 * a2';
Z = sigmoid(thetaa2Trans); %10*5000

[high_val,row_id] = max(Z);

%hyp is not required 
hyp = Z==high_val;

sizeHyp = size(hyp); %10*5000
%%%%
%hyp(:,1:5) will be like below
%
%  0  0  0  0  0
%  0  0  0  0  0
%  0  0  0  0  0
%  0  0  0  0  0
%  0  0  0  0  0
%  0  0  0  0  0
%  0  0  0  0  0
%  0  0  0  0  0
%  0  0  0  0  0
%  1  1  1  1  1


Y = zeros(num_labels,length(y));
sizeY = size(Y); %10*5000

for  i = 1:length(y)
  Y(y(i),i) = 1;
end;

% if y is [1;2;3;4;5;6;7;8;9]
% then Y will become
% 1  0   0   0   0   0   0   0   0
%  0    1  0   0   0   0   0   0   0
%  0   0    1  0   0   0   0   0   0
%  0   0   0    1  0   0   0   0   0
%  0   0   0   0    1  0   0   0   0
%  0   0   0   0   0    1  0   0   0
%  0   0   0   0   0   0    1  0   0
%  0   0   0   0   0   0   0    1  0
%  0   0   0   0   0   0   0   0    1

term1InCostFunction = -Y.*log(Z);
term2InCostFunction = (1-Y).*log(1-Z);

J = sum(sum(term1InCostFunction-term2InCostFunction))/m;




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
smalldelta3 = Z-Y; %10*5000

%%%%%
%%%%%
%IMPORTANT: MIND THAT IN sigmoidGradient you should pass the values before applying sigmoid function, else it wont work.
%%%%%
%%%%%
smalldelta2 = (Theta2' * smalldelta3)' .* sigmoidGradient([ones(size(Xforlayer3BeforeSigmoid,1),1), Xforlayer3BeforeSigmoid]); %5000*26

delta1 = X' * smalldelta2;
delta1 = delta1'; %26*401
%%%%%
%%%%%
%This is important to discard the 1 feature term for each hidden layer (only for hidden layer)
%%%%%
%%%%%
delta1 = delta1(2:end,:); %discarding errors calulated for +1 feature (1st component in hidden layer). smalldelta size should be equal to theta1_grad
delta2 = a2' * smalldelta3';
delta2 = delta2'; %10*26

%Theta1_grad = delta1/m;  %Without Rationalization.
Theta1_grad(:,1) = delta1(:,1)/m;
Theta1_grad(:,2:end) = delta1(:,2:end)/m + (lambda*Theta1(:,2:end))/m;

%Theta2_grad = delta2/m; %Without Rationalization
Theta2_grad(:,1) = delta2(:,1)/m;
Theta2_grad(:,2:end) = delta2(:,2:end)/m + (lambda*Theta2(:,2:end))/m;
 

 
 

  
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%adding regularization to cost function
regularizableThetas = [Theta1(:,2:end)(:); Theta2(:,2:end)(:);];

squaredRegularizableThetas = regularizableThetas.^2;
J = J+ (sum(squaredRegularizableThetas)*lambda)/(2*m);

















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
