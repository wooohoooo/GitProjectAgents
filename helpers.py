
import numpy as np

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


from sklearn.model_selection import train_test_split
import pandas as pd

def test_CoBEAU_NLPD_(model_type,model_params,X,y,num_experiments=100,plot=True):
	results_list = []
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
	np.random.seed(42)
	for i in range(num_experiments):
		model = model_type(**model_params)
		model.fit(X_train,y_train)
		results = model.self_evaluate(X_test,y_test)
		results_list.append(results)
		np.random.seed(42+i)

	df_results = pd.DataFrame.from_records(results_list)
	if plot:
		model.mutli_dimenstional_scatterplot(X_test,y_test,figsize=(20,5))
		
	return df_results

def test_CoBEAU_NLPD_mixed(model_list,X,y,num_experiments=100):
	for model in model_list:
		pass
		
		
import matplotlib.pyplot as plt
		
def normalized_plot(df):
	normalized_df=(df-df.mean())/df.std()
	plt.plot(normalized_df['NLPD'])
	plt.plot(normalized_df['correlation between error and variance'])