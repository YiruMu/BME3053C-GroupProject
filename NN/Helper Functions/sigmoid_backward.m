function [dA,cache]= sigmoid_backward(dA, cache)
    %Implement the backward propagation for a single SIGMOID unit.

    %Arguments:
    %dA -- post-activation gradient, of any shape
    %cache -- 'Z' where we store for computing backward propagation efficiently

    %Returns:
    %dZ -- Gradient of the cost with respect to Z
    
    
    Z = cache;
    
    s = 1./(1+exp(-Z));
    dZ = dA .* s .* (1-s);
    
   % assert (size(dZ) == size(Z));
    
    
end