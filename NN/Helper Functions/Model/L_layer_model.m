function   parameters = L_layer_model(X, Y, layers_dims, learning_rate, num_iterations, print_cost)
    
    %Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
    
    %Arguments:
    % X -- data, array of shape 
    % Y -- true "label" vector 
    % layers_dims -- list containing the input size and each layer size, of
    % length. 
    % learning_rate -- learning rate of the gradient descent update rule
    % num_iterations -- number of iterations of the optimization loop
    % print_cost -- if True, it prints the cost every 5 steps
    
    % Returns:
    % parameters -- parameters learnt by the model. They can then be used to predict.
   

    rng(1);
    costs = []   ;                      % keep track of cost
    
    % Parameters initialization. 
   
    parameters = initialize_parameters_deep(layers_dims)
   
    
    % Loop (gradient descent)
    for i = 1: num_iterations

        % Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        
        [AL, caches] = L_model_forward(X,parameters)
        
        
        % Compute cost.
        cost = compute_cost(AL,Y)
        
    
        %  Backward propagation.
        grads = L_model_backward(AL,Y,caches)
        
 
        % Update parameters.
        parameters = update_parameters(parameters,grads,learning_rate)
        
                
        %  Print the cost every 5 training example
        if ( print_cost && mod(i,5) ==0 )
            fprintf ("Cost after iteration %f: %f", i, cost);
        end 
        if (print_cost && mod(i,5) == 0) 
            costs = [costs cost];
        end 
    end         
    %plot the cost
   plot(squeeze(costs))
   ylabel('cost')
   xlabel('iterations (per 5th)')
   title("Learning rate =" + str(learning_rate))
   
    
 end 