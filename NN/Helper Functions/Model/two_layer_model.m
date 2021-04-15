function parameters = two_layer_model(X, Y, layers_dims, learning_rate, num_iterations, print_cost)

    %Implements a two-layer neural network: LINEAR->RELU->LINEAR->SIGMOID.
    
    %Arguments:
    %X -- input data, of shape (n_x, number of examples) p.s. flipped 
    %Y -- true "label" vector (containing 1 if COVID patient, 0 if non-COVID patient), of shape ( 1,number of examples)
    %layers_dims -- dimensions of the layers (n_x, n_h, n_y)
    %num_iterations -- number of iterations of the optimization loop
    %learning_rate -- learning rate of the gradient descent update rule
    %print_cost -- If set to True, this will print the cost every 5 iterations 
    
    %Returns:
    %parameters -- a dictionary/map containing W1, W2, b1, and b2
  
    
    rng(1) %random seed of 1 
    costs = [];                              % to keep track of the cost
    m = size(X,2);    % number of examples
    
    
     n_x = layers_dims(1);
     n_h = layers_dims(2);
     n_y = layers_dims(3);
    %[n_x, n_h, n_y] = layers_dims; 
    
    % Initialize parameters map, by calling the helper functions 
    parameters = initialize_parameters(n_x,n_h,n_y);
    
    
    % Get W1, b1, W2 and b2 from the map parameters.
    W1 = parameters('W1');
    b1 = parameters('b1');
    W2 = parameters('W2');
    b2 = parameters('b2');
    
    % Loop (gradient descent)

    for i = 1:num_iterations

        %Forward propagation: LINEAR -> RELU -> LINEAR -> SIGMOID. Inputs: "X, W1, b1, W2, b2". Output: "A1, cache1, A2, cache2".
        [A1, cache1] = linear_activation_forward(X,W1,b1,"relu");
        [A2, cache2] = linear_activation_forward(A1,W2,b2,"sigmoid");
        
        
        %Compute cost
        cost = compute_cost(A2,Y);
        
        % Initializing backward propagation
        dA2 = - (Y./ A2) - (1 - Y)./( 1 - A2);
        
        %Backward propagation. Inputs: "dA2, cache2, cache1". Outputs: "dA1, dW2, db2; also dA0 (not used), dW1, db1".
        [dA1, dW2, db2] = linear_activation_backward(dA2,cache2,"sigmoid");
        [dA0, dW1, db1] = linear_activation_backward(dA1,cache1,"relu");
        
        
        % Set grads['dWl'] to dW1, grads['db1'] to db1, grads['dW2'] to dW2, grads['db2'] to db2
        keys = {'dW1', 'db1', 'dW2','db2'};
        values = {dW1, db1, dW2, db2};
        grads = containers.Map (keys, values);
        % note: grads will be a map 
        
        % Update parameters.
        
        parameters = update_parameters(parameters,grads,learning_rate);
        

        %Retrieve W1, b1, W2, b2 from parameters
        W1 = parameters('W1');
        b1 = parameters('b1');
        W2 = parameters('W2');
        b2 = parameters('b2');
        
        %Print the cost every 5 training example
        if (print_cost && mod(i,5) == 0)
            fprintf("Cost after iteration %f: %f",i, squeeze(cost));
        end 
        if (print_cost && mod(i,5)==0)
            costs = [costs cost];
        end 
     end  
    %plot the cost

    plot(squeeze(costs));
    ylabel('cost')
    xlabel('iterations (per 5th)')
    title("Learning rate =" + num2str(learning_rate))
    shg;
    
   
end