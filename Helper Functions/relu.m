function [A, cache] = relu(Z)
    %Implement the RELU function.

    %Arguments:
    %Z -- Output of the linear layer, of any shape

    %Returns:
    %A -- Post-activation parameter, of the same shape as Z
    %cache -- a MATLAB map(i.e. python dictionary) containing "A" ; stored for computing the backward pass efficiently
   

    A = max(0,Z);
    if (A.size ~= Z.size)
        disp('Something is wrong with relu function');
    end 
    
    cache = Z;
    
end 