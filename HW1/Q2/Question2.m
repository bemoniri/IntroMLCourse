clc
clear all
close all

%% Creating Data Matrix
M = csvread('train.csv', 1);
x_train = M(:,1);
y_train = M(:,2);

M = csvread('validation.csv', 1);
x_test = M(:,1);
y_test = M(:,2);

clear M

%% Fitting Model
clear Emp_Error True_Error
figure
for p = 1 : 14
    X_Train = [];
    X_Test = [];
    for k = 0 : p
       X_Train = [X_Train, x_train.^k];
       X_Test = [X_Test, x_test.^k];
    end
    %omega = (X_Train'*X_Train)\(X_Train'*y_train);
    omega = regress(y_train, X_Train);
    subplot(5,3,p)
    hold on
    plot(x_train, y_train, '.')
    plot(x_train, X_Train*omega, '*')
    title(['Degree = ', num2str(p)])
    Emp_Error(p) = sum((y_train - X_Train*omega).^2)/length(y_train);
    True_Error(p) = sum((y_test - X_Test*omega).^2)/length(y_test);
    xlabel('x');
    ylabel('y')
end

figure
hold on
plot(Emp_Error)
plot(True_Error)
title('True Error and Empirical Error w.r.t. Polynomial Degree')
legend('Empirical Risk', 'True Risk')
xlabel('Poly Degree')
ylabel('Risk')

