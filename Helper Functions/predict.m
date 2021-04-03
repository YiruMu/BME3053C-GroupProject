function p = predict(X,y,parameters)
    %This function is used to predict the results of a  L-layer neural network.
    
    %Arguments:
    %X -- data set of examples you would like to label
    %parameters -- parameters of the trained model
    
    %Returns:
    %p -- predictions for the given dataset X
    
    
    m = X.size(2);
    n = length(parameters); % 2 # number of layers in the neural network
    p = zeros(1,m);
    
    %Forward propagation
    [probas, caches] = L_model_forward(X, parameters);

    
    %convert probas to 0/1 predictions
    for i =1: probas.size(2) 
        if (probas(1,i) > 0.5)
            p(1,i) = 1;
        else
            p(1,i) = 0;
        end 
    end 
    
    %print results
    %print ("predictions: " + num2str(p))
    %print ("true labels: " + num2str(y))
    fprintf("Accuracy: "  + num2str(sum((p == y)/m))); 
        
end
