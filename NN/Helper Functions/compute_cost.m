function cost = compute_cost(AL,Y)
    %Implement the cost function defined by equation (7).

    %Arguments:
    %AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    %Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    %Returns:
    %cost -- cross-entropy cost
    
    m = size(Y,2);

    % Compute loss from aL and y.
    cost = (1/m)*sum(sum(-Y.*log(AL)-(1-Y).*log(1-AL)));
    %cost = 1/m*sum(sum(-y_matrix.*log(a3)-(1-y_matrix).*log(1-a3)));
    
    cost = squeeze(cost);      % To make sure cost's shape is what we expect (e.g. this turns [[17]] into 17).
    %assert(size(cost) == [][]);
    
end 