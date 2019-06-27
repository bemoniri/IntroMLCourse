function [w] = perceptron(X,Y,w_init, numsteps)

w = w_init;
for iteration = 1 : numsteps  
  for ii = 1 : size(X,2)      
    if sign(w'*X(:,ii)) ~= Y(ii)
      w = w + X(:,ii) * Y(ii);
      break;
    end
  end
end
%Normalize
w = w / norm(w);