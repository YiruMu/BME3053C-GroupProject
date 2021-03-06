function grads = L_model_backward(AL, Y, caches)
    
    %Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
    
    
    %Arguments:
    %AL -- probability vector, output of the forward propagation (L_model_forward())
    %Y -- true "label" vector (containing 0 if non-COVID, 1 if COVID)
    %caches -- list of caches containing:
                %every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                %the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
    
    %Returns:
    %grads -- A map with the gradients
             %grads["dA" + str(l)] = ... 
             %grads["dW" + str(l)] = ...
             %grads["db" + str(l)] = ... 
    
    grads = {}; 
    L = length(caches); % the number of layers
    m = size(AL,2);
    Y = reshape(Y, size(AL)); % after this line, Y is the same shape as AL
    
    % Initializing the backpropagation
    dAL = -(Y./AL)-(1-Y)./(1-AL);
   
    
    % Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "dAL, current_cache". Outputs: "grads["dAL-1"], grads["dWL"], grads["dbL"]
    
    temp1 = caches{L-1};
    temp2 = caches{L}
    tempcurrent_cache = containers.Map({'linear_cache','activation_cache'},{temp1,temp2}); 
    [tempdA, tempdW, tempdb] = linear_activation_backward(dAL,tempcurrent_cache,"sigmoid");
    grads(strcat('dA', num2str(L))) = tempdA;
    grads(strcat('dW', num2str(L-1))) = tempdW;
    grads(strcat('db', num2str(L-1))) = tempdb;
    
    % Loop from l=L-2 to l=0
    for i = L:-1:2
        % lth layer: (RELU -> LINEAR) gradients.
        % Inputs: "grads["dA" + str(l + 1)], current_cache". Outputs: "grads["dA" + str(l)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)] 
        
        temp1 = caches{i-1};
        temp2 = caches{i}
        current_cache = containers.Map({'linear_cache','activation_cache'},{ temp1,temp2});
        [dA_prev_temp, dW_temp, db_temp] = linear_activation_backward(grads(strcat('dA',num2str(i))),current_cache,"relu");
        grads(strcat('dA',num2str(i-1))) = dA_prev_temp;
        grads(strcat('dW', num2str(i))) = dW_temp;
        grads(strcat('db', num2str(i))) = db_temp;
        
    end 
end 
