function [Z, cache] = linear_forward(A,W,b)
    %Implement the linear part of a layer's forward propagation.

    %Arguments:
    %A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    %W -- weights matrix: array of shape (size of current layer, size of previous layer)
    %b -- bias vector, array of shape (size of the current layer, 1)

    %Returns:
    %Z -- the input of the activation function, also called pre-activation parameter 
    %cache -- a MATLAB container map containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    
    
    Z = W*A+b;
    
   
    keySet = {'A','W','b'};
    valueSet = {A, W, b};
    cache = containers.Map(keySet,valueSet);  % MATLab doesn't have tuple
    
    
    
end 