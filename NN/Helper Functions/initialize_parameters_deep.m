function parameters = initialize_parameters_deep(layer_dims)
% Arguments:
% layer_dims --  array (list) containing the dimensions of each layer in our network
    
% Returns:
% parameters -- matlab container map containing your parameters "W1", "b1", ..., "WL", "bL":
               % Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
               % bl -- bias vector of shape (layer_dims[l], 1)
   
               
rng(1)

L = length(layer_dims);            % number of layers in the network
parameters = containers.Map;
for i = 1:L-1 
  parameters(strcat('W', num2str(i))) = randn(layer_dims(i+1), layer_dims(i)) / sqrt(layer_dims(i))%*0.01
  parameters(strcat('b',num2str(i))) = zeros(layer_dims(i+1), 1)
        

end

