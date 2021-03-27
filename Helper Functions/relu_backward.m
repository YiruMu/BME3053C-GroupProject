function dZ = relu_backward(dA, cache)

    %Implement the backward propagation for a single RELU unit.

    %Arguments:
    %dA -- post-activation gradient, of any shape
    %cache -- 'Z' where we store for computing backward propagation efficiently

    %Returns:
    %dZ -- Gradient of the cost with respect to Z
    
    Z = cache;
    dZ = dA; %just converting dz to a correct object. dZ is a copy of dA not a reference
    % When z <= 0, you should set dz to 0 as well. 
    dZ(Z <= 0)= 0;
    
    if (dZ.size ~= Z.size)
        disp('Something is wrong with the relu function');
    end 
    
end 