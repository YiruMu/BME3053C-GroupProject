function [dA_prev, dW, db] = linear_backward(dZ,cache)
%Implement the linear portion of backward propagation for a single layer (layer l)
%Arguments:
    %dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    %cache -- map of values (A_prev, W, b) coming from the forward propagation in the current layer

    %Returns:
    %dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    %dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    %db -- Gradient of the cost with respect to b (current layer l), same shape as b
    [A_prev, W, b] = values(cache);
    m = size(A_prev,2);

    
    dW = 1/m*(dZ * A_prev');
    db = 1/m*sum(dZ,2);
    dA_prev = W'*dZ;
   
    
    assert (size(dA_prev) == size(A_prev))
    assert (size(dW) == size(W))
    assert (size(db) == size(b))
end

