import pandas
from sklearn import neural_network, preprocessing

def mae_func(preds, actuals):
	num_preds = len(preds)
	resids = 0
	for i in range(0, num_preds):
		resids += abs(preds[i] - actuals[i])
	mean_error = resids/num_preds
	return(mean_error)

walmart_sales_df = pandas.read_csv("~/Desktop/walmart_sales_model.csv", header=None)

walmart_sales_data = walmart_sales_df.values

walmart_sales_actuals = (walmart_sales_data[:,0])

walmart_sales_preds = preprocessing.scale(walmart_sales_data[:,1:])

network_hidden_layers = (120, 70, 70, 50, 50, 30, 30, 10, 5)

base_network = neural_network.MLPRegressor(network_hidden_layers, activation = 'relu', alpha = .005) # creating a neural network with tanh activation function

network_trained_model = base_network.fit(walmart_sales_preds, walmart_sales_actuals)

walmart_test_df = pandas.read_csv("~/Desktop/walmart_test_data.csv", header=None)

walmart_testing_data = walmart_test_df.values

walmart_testing_actuals = walmart_testing_data[:,0]

walmart_testing_predictors = preprocessing.scale(walmart_testing_data[:,1:])

predictions = network_trained_model.predict(walmart_testing_predictors)

network_error = mae_func(predictions, walmart_testing_actuals)

print network_error
