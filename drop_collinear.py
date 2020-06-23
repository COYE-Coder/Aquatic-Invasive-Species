"""
Function that will drop any collinear features given a Dataframe, 
choosing the first one to drop
inputs: 
	df -- Data frame containing features of interest
	Threshold -- Pearson's correlation threshold of features to drop

"""





import numpy as np


def drop_collinear(df,threshold):

	# Create correlation matrix
	corr_matrix = df.corr().abs()

	# Select upper triangle of correlation matrix
	upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

	# Find features with correlation greater than threshold
	to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

	# Drop features 
	df.drop(to_drop, axis=1, inplace=True)

	return df



