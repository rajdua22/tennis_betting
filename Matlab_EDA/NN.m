%% Machine Learning Tennis Research

% Rajvir Dua
% Basic Neural NEtwork Using Regression

%% Initialization
clear ; close all; clc

load('Data.mat');
Surface = Data(:,7);
Sets = Data(:,9);
Wrank = Data(:,12);
Lrank = Data(:,13);

Surface = table2array(Surface);
Surface = double(Surface);
Surface = Surface - 1;

Sets = table2array(Sets);
Sets = Sets == 3;

Wrank = table2array(Wrank);
Lrank = table2array(Lrank);
max2 = max(Wrank, [], 1, 'omitnan');
max3 = max(Lrank, [], 1, 'omitnan');
if max3>max2
    max2 = max3;
end

% mean2 = mean(Wrank, 'omitnan');
% Wrank = Wrank - mean2;
Wrank = Wrank ./ max2;

% mean2 = mean(Lrank, 'omitnan');
% Lrank = Lrank - mean2;
Lrank = Lrank ./ max2;



m = size(Wrank,1);
data = [Surface Sets Wrank Lrank ones(m,1)];
data = rmmissing(data);
X = data(:,1:4);
m = size(data,1);

y = X(:,3) < X(:,4);
temp = X(:,3);
temp2 = X(:,4);
for i = 1:m
    if y(i) == 0
        temp(i) = X(i,4);
        temp2(i) = X(i,3);
    end
end
X(:,3) = temp;
X(:,4) = temp2;
X(:,5) = X(:,4) - X(:,3);
% y = 1 - y;


% y = data(:,5);
% mid = m / 2;
% temp = X(mid:end,3);
% X(mid:end, 3) = X(mid:end, 4);
% X(mid:end, 4) = temp;
% y(mid:end) = 0;
y = y + 1;



[m, n] = size(X);
% X = [ones(m, 1) X];
initial_theta = zeros(n + 1, 1);
train = m * 0.8;
Xtrain = X(1:train,:);
Ytrain = y(1:train,:);
Xtest = X(train:end, :);
Ytest = y(train:end,:);

%% Setup the parameters 
input_layer_size  = 5;  
hidden_layer_size = 5;   
num_labels = 2;          

fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];


%% =============== Part 7: Implement Backpropagation ===============
%  Once your cost matches up with ours, you should proceed to implement the
%  backpropagation algorithm for the neural network. You should add to the
%  code you've written in nnCostFunction.m to return the partial
%  derivatives of the parameters.
%
fprintf('\nChecking Backpropagation... \n');

%  Check gradients by running checkNNGradients
checkNNGradients;

fprintf('\nProgram paused. Press enter to continue.\n');
pause;


%% =================== Part 8: Training NN ===================
%  You have now implemented all the code necessary to train a neural 
%  network. To train your neural network, we will now use "fmincg", which
%  is a function which works similarly to "fminunc". Recall that these
%  advanced optimizers are able to train our cost functions efficiently as
%  long as we provide them with the gradient computations.
%
fprintf('\nTraining Neural Network... \n')

%  After you have completed the assignment, change the MaxIter to a larger
%  value to see how more training helps.
options = optimset('MaxIter', 400);

%  You should also try different values of lambda
lambda = 0;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, Xtrain, Ytrain, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

fprintf('Program paused. Press enter to continue.\n');
pause;


%  After training the neural network, we would like to use it to predict
%  the labels. You will now implement the "predict" function to use the
%  neural network to predict the labels of the training set. This lets
%  you compute the training set accuracy.

pred = predictNN(Theta1, Theta2, Xtest);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == Ytest)) * 100);


