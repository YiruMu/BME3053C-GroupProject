function [AL, caches] = L_model_forward(X, parameters)
%Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
    
   % Arguments:
   %X -- data, array of shape (input size, number of examples)
   %parameters -- output of initialize_parameters_deep()
    
   %Returns:
   %AL -- last post-activation value
   %caches -- list of caches containing:
                %every cache of linear_activation_forward() (there are L-1 of them, indexed from 0 to L-1)
    

    
    A = X;
    L = length(parameters); % 4                 # number of layers in the neural network
    
    % Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    for i=1:L/2-1
        A_prev = A; 
        W = parameters(strcat('W',num2str(i)));
        b = parameters(strcat('b',num2str(i)));
        [A, cache] = linear_activation_forward(A_prev,W,b,"relu");
        
        linearCache = cache('linear_cache');
        activationCache = cache('activation_cache');
        
        caches = [caches; linearCache, activationCache]; 
    end 
    
    % Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
   
    [AL, cache] = linear_activation_forward(A,parameters(strcat('W',num2str(L/2))),parameters(strcat('b',num2str(L/2))),"sigmoid");
    % need to be modified, 2 is hard coded for now 
    linearCache = cache('linear_cache');
    activationCache = cache('activation_cache');
     caches = [caches; linearCache, activationCache]; 
    
    
   % assert(size(AL) == [1,size(X,1)]);
            
end 