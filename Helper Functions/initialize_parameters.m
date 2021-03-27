function parameters = initialize_parameters(n_x, n_h, n_y)
    %Argument:
    %n_x -- size of the input layer
    %n_h -- size of the hidden layer
    %n_y -- size of the output layer
    
    %Returns:
    %parameters -- MATLAB container map containing your parameters:
                    %W1 -- weight matrix of shape (n_h, n_x)
                    %b1 -- bias vector of shape (n_h, 1)
                    %W2 -- weight matrix of shape (n_y, n_h)
                    %b2 -- bias vector of shape (n_y, 1)
    
    
    rng(1);
    W1 = randn(n_h,n_x).*0.01;
    b1 = zeros(n_h,1);
    W2 = randn(n_y,n_h).*0.01;
    b2 = zeros(n_y,1);
   
    
    assert(W1.size == [n_h, n_x]);
    assert(b1.size == [n_h, 1]);
    assert(W2.size == [n_y, n_h]);
    assert(b2.size == [n_y, 1]);
    
    keySet = {'W1','b1','W2','b2'};
    valueSet = [W1,b1,W2, b2];
    parameters = containers.Map(keySet,valueSet);
    
end 

% using map containers
% https://www.mathworks.com/help/matlab/map-containers.html 