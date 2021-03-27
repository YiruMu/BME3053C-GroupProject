function [Z, cache] = linear_forward(A,W,b)
    %Implement the linear part of a layer's forward propagation.

    %Arguments:
    %A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    %W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    %b -- bias vector, numpy array of shape (size of the current layer, 1)

    %Returns:
    %Z -- the input of the activation function, also called pre-activation parameter 
    %cache -- a MATLAB container map/python tuple containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    
    
    Z = (W*A)+b;
    
    assert(Z.size == [W.size(1), A.size(2)]);
    keySet = {'A','W','b'};
    valueSet = [A, W, b];
    cache = container.Map(keySet,valueSet);  % MATLab doesn't have tuple
    
    
    
end 