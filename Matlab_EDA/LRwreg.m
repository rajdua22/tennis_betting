%% Machine Learning Tennis Research Project
%
%  Rajvir Dua
%  ------------
% Logistic Regression

%% Initialization
clear ; close all; clc

%% Load Data
load('Data.mat');
Outdoor = Data(:,6);
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
m = size(data,1);
X = data(:,1:4);



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
% y = 1 - y;

% y = data(:,5);
% mid = m / 2;
% temp = X(mid:end,3);
% X(mid:end, 3) = X(mid:end, 4);
% X(mid:end, 4) = temp;
% y(mid:end) = 0;



[m, n] = size(X);
X = [ones(m, 1) X];
initial_theta = zeros(n + 1, 1);
train = m * 0.8;
Xtrain = X(1:train,:);
Ytrain = y(1:train,:);
Xtest = X(train:end, :);
Ytest = y(train:end,:);


lambda = 0;
[cost, grad] = costFunctionReg(initial_theta, Xtrain, Ytrain,lambda);
fprintf('Cost at initial theta (zeros): %f\n', cost);
fprintf('Gradient at initial theta (zeros): \n');
fprintf(' %f \n', grad);
% test_theta = [30; 0; 0.2; 0.2; 0.2];
% [cost, grad] = costFunction(test_theta, X, y);
% fprintf('\nCost at test theta: %f\n', cost);
% fprintf('Gradient at test theta: \n');
% fprintf(' %f \n', grad);


options = optimset('GradObj', 'on', 'MaxIter', 8000);
 [theta, J, exit_flag] = ...
	fminunc(@(t)(costFunctionReg(t, X, y, lambda)), initial_theta, options);
fprintf('Cost at theta found by fminunc: %f\n', cost);
fprintf('theta: \n');
fprintf(' %f \n', theta);


player1 = 50 / max2;
player2 = 1/ max2;
prob = sigmoid([1 3 1 player1 player2] * theta);
fprintf(['For player1 with a rank of 1 and player2 with a rank of 200 (5 sets on hardcourt), we predict an win ' ...
         'probability of %f\n'], prob);
p = predict(theta, Xtest);
fprintf('Train Accuracy: %f\n', mean(double(p == Ytest)) * 100);