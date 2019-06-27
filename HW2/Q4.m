%% Checking My Perceptron Algorithm
clc
clear

X = rand(2, 100);
for i = 1 : 100
Y(i) = sign(0.4*X(1,i)-0.5*X(2,i));
end

Xp = X(:, Y>0);
Xm = X(:, Y<0);
w = perceptron(X, Y, [0 0]', 10000);
yline = -w(1)/w(2)*(0:0.01:1);

figure
hold on
plot(Xm(1,:), Xm(2,:), '.')
plot(Xp(1,:), Xp(2,:), '.')
plot(0:0.01:1, yline)
xlabel('X'); ylabel('Y')
title('Testing the Perceptron Algorithm')
legend('Label = 1', 'Label = 2', 'Perceptron')
T = [X; Y];

%% Loading and Visualizing Data
clc
clear

addpath('../npy-matlab-master/npy-matlab');
X_train = double(readNPY("X_train.npy"));
X_test = double(readNPY("X_test.npy"));
Y_train = double(readNPY("Y_train.npy"));
Y_test = double(readNPY("Y_test.npy"));

X_train = [X_train, ones(size(X_train, 1), 1)];
X_test = [X_test, ones(size(X_test, 1), 1)];
figure
hold on
plot3(X_train(Y_train==1,1),X_train(Y_train==1,2),X_train(Y_train==1,3),'.')
plot3(X_train(Y_train==-1,1),X_train(Y_train==-1,2),X_train(Y_train==-1,3),'.')
xlabel('X'); ylabel('Y'); zlabel('Z')
legend('y=1', 'y=-1')
title('Scatter Plot of the Data')


%% Risk for Linear Classification
perceptron(X_train', Y_train', [0 0 0 0]', 10000)

w = [0 0 0 0]';
loss = [];

for iter = 1 : 20
    w = perceptron(X_train', Y_train', [0 0 0 0]', 500*iter);
    loss(iter) = sum(sign(w'*X_test')~=Y_test')/500
end

w 

figure
plot(500*(1:20), loss)
title('True Risk');
ylabel('Tisk');
xlabel('Iterations')

%% Adding z^3
clear
X_train = double(readNPY("X_train.npy"));
X_test = double(readNPY("X_test.npy"));
Y_train = double(readNPY("Y_train.npy"));
Y_test = double(readNPY("Y_test.npy"));

X_train = [X_train, X_train(:,3).^3, ones(size(X_train, 1), 1)];
X_test = [X_test, X_test(:,3).^3, ones(size(X_test, 1), 1)];

perceptron(X_train', Y_train', [0 0 0 0 0 ]', 10000)

w = [0 0 0 0]';
loss = [];

for iter = 1 : 20
    w = perceptron(X_train', Y_train', [0 0 0 0 0]', 500*iter);
    loss(iter) = sum(sign(w'*X_test')~=Y_test')/500
end

w

figure
plot(500*(1:20), loss)
title('True Risk for Cubic Perceptron');
ylabel('Tisk');
xlabel('Iterations')


%% SVM linear

addpath('../npy-matlab-master/npy-matlab');
X_train = double(readNPY("X_train.npy"));
X_test = double(readNPY("X_test.npy"));
Y_train = double(readNPY("Y_train.npy"));
Y_test = double(readNPY("Y_test.npy"));

model = fitcecoc(X_train, Y_train')
disp('done')
loss = sum(predict(model, X_test) ~= Y_test)/500
train_loss = sum(predict(model, X_train) ~= Y_train)/2000

omega = model.BinaryLearners{1,1}.Beta

%% SVM Cubic


addpath('../npy-matlab-master/npy-matlab');
X_train = double(readNPY("X_train.npy"));
X_test = double(readNPY("X_test.npy"));
Y_train = double(readNPY("Y_train.npy"));
Y_test = double(readNPY("Y_test.npy"));

X_train = [X_train, X_train(:,3).^3];
X_test = [X_test, X_test(:,3).^3];

model = fitcecoc(X_train, Y_train')
disp('done')
loss = sum(predict(model, X_test) ~= Y_test)/500
train_loss = sum(predict(model, X_train) ~= Y_train)/2000

omega = model.BinaryLearners{1,1}.Beta