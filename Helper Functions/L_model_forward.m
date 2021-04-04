function [AL, caches] = L_model_forward(X, parameters)
%Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
    
   % Arguments:
   %X -- data, numpy array of shape (input size, number of examples)
   %parameters -- output of initialize_parameters_deep()
    
   %Returns:
   %AL -- last post-activation value
   %caches -- list of caches containing:
                %every cache of linear_activation_forward() (there are L-1 of them, indexed from 0 to L-1)
    

    caches = [];
    A = X;
    L = length(parameters); % 2                  # number of layers in the neural network
    
    % Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    for i=1:L 
        A_prev = A; 
        [A, cache] = linear_activation_forward(A,parameters('W'+num2str(i)),parameters('b'+num2str(i)),'relu');
        caches.append(cache);
    end 
    
    % Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
   
    [AL, cache] = linear_activation_forward(A,parameters('W'+num2str(L)),parameters('b'+num2str(L)),'sigmoid');
    caches.append(cache);
    
    
    assert(size(AL) == [1,size(X,2)]);
            
end 