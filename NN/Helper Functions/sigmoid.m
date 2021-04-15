function [A, cache] = sigmoid(Z)
  
    %Implements the sigmoid activation 
    
    %Arguments:
    %Z --  array of any shape
    
    %Returns:
    %A -- output of sigmoid(z), same shape as Z
    %cache -- returns Z as well, useful during backpropagation
  
    
    A = 1./(1+exp(-1.*Z));
    cache = Z;
    
end 