clc
clear all
close all
format long

%% Seperating Test and Train Data
data = csvread('data_Q1.csv', 1);
data_x = data(:,1:8);
data_y = data(:,9);

%I = randperm(length(data_y));
I = 1:length(data_y);
val_size = ceil(length(data_y)/4);

X_Val = data_x(I(1:val_size),:);
Y_Val = data_y(I(1:val_size),:);

X_Train = data_x(I(val_size+1:end),:);
Y_Train = data_y(I(val_size+1:end),:);

X_Train = [X_Train, ones(size(X_Train, 1),1)]';
X_Val = [X_Val, ones(size(X_Val, 1),1)]';

clear data data_x data_y

%% Part One
figure
for feature = 1 : 8
    subplot(4,2,feature)
    hold on
    plot(X_Train(feature, :), Y_Train, '.');
    plot(X_Val(feature, :), Y_Val, '.');
    xlabel(['Feature ', num2str(feature)])
    ylabel('y')
    title(['Scatter Plot for Feature ', num2str(feature)])
    legend('Train', 'Test')
end

%% Part Two

omega = inv(X_Train*X_Train')*X_Train*Y_Train

Yhat_Train = X_Train'*omega;
Yhat_Val = X_Val'*omega;
Val_Loss = sum((Yhat_Val - Y_Val).^2)/length(Y_Val)
Train_Loss = sum((Yhat_Train - Y_Train).^2)/length(Y_Train)

