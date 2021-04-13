function parameters = update_parameters(parameters, grads, learning_rate)
%Update parameters using gradient descent
    %Arguments:
    %parameters -- matlab map containing your parameters 
    %grads -- matlab map containing your gradients, output of L_model_backward
    
    %Returns:
    %parameters -- matlab map containing your updated parameters 
                  %parameters["W" + str(l)] = ... 
                  %parameters["b" + str(l)] = ...
  
    
    for i = 1:length(parameters)/2
        % parameters(strcat('W',num2str(i)))-learning_rate*grads(strcat('dW',num2str(i)))};
          wi =  parameters(strcat('W',num2str(i)));
          dwi = grads(strcat('dW',num2str(i)));
        parameters(strcat('W',num2str(i))) = wi -learning_rate*dwi;
        % parameters(strcat('b',num2str(i)))-learning_rate*grads(strcat('db',num2str(i)))
        bi = parameters(strcat('b',num2str(i)));
        dbi = grads(strcat('db',num2str(i)));
        parameters(strcat('b',num2str(i))) = bi-learning_rate*dbi;
    end 
end 

