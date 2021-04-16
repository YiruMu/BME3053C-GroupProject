function p = predict(X,y,parameters)
    %This function is used to predict the results of a  L-layer neural network.
    
    %Arguments:
    %X -- data set of examples you would like to label
    %parameters -- parameters of the trained model
    
    %Returns:
    %p -- predictions for the given dataset X
    
    
    m = size(X,2);
    n = length(parameters)/2; % 2 # number of layers in the neural network
    p = zeros(1,m);
    
    %Forward propagation
    [probas, ~] = L_model_forward(X, parameters);
    
    
    %convert probas to 0/1 predictions
    for i =1: size(probas,2) 
        if (probas(1,i) > 0.7)
            p(1,i) = 1;
        else
            p(1,i) = 0;
        end 
    end 
    
    %print results
    %print ("predictions: " + num2str(p))
    %print ("true labels: " + num2str(y))
    fprintf("Accuracy: %f", sum((p == y)/m)); 
        
end

