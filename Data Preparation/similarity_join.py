# similarity_join.py
# Running Assumptions - This code is prepared assuming that both .csv datasets should be present in the same folder/root folder where source code is present.
import re
import pandas as pd
import numpy as np
import sys
from sqlalchemy import create_engine
engine = create_engine('sqlite://', echo=False)

class SimilarityJoin:
    def __init__(self, data_file1, data_file2):
        self.df1 = pd.read_csv(data_file1)
        self.df2 = pd.read_csv(data_file2)

    def preprocess_df(self, df, cols):
        # Function to split, strip and remove nan from words contained in concatenated columns
        def tokenize(string):
            token = re.split(r'\W+', string.lower())
            token = list(filter(str.strip, token))
            token = [item for item in token if str(item) != 'nan']
            return token

        # Combine the columns in scope
        df["joinKey"] = df[cols[0]].astype(str) + ' ' + df[cols[1]].astype(str)
        # Tokenize the words
        df["joinKey"] = df["joinKey"].map(tokenize)
        return df

    def filtering(self, df1, df2):
        #Explode the listed values in joinKey column to equivalent rows
        df1_exp = df1.explode("joinKey")
        df2_exp = df2.explode("joinKey")
        #Renaming column joinKey in both dataframe for further usage
        df1.rename(columns={'joinKey': 'joinKey1'}, inplace=True)
        df2.rename(columns={'joinKey': 'joinKey2'}, inplace=True)
        #Selecting only id and joinKey as they are the columns in scope of our interest
        df1 = df1[['id','joinKey1']]
        df2 = df2[['id','joinKey2']]

        #Creating temporary table for performing SQL query
        df1_exp.to_sql("df1exp", con=engine)
        df2_exp.to_sql("df2exp", con=engine)

        #Query to retrieve non duplicate ids from both the exploded dfs based on matching joinKeys
        new_list = engine.execute("SELECT DISTINCT df1.id,df2.id FROM df1exp as df1,df2exp as df2 WHERE df1.joinKey = df2.joinKey").fetchall()
        new_df = pd.DataFrame(list(new_list),columns=['id1','id2'])

        #Query to retrieve the data from the initial dfs based on matching ids. Dataframe new_df is joined with df1 and df2 to obtain the result
        cand_df = new_df.join(df1.set_index('id'),on='id1',how='inner').join(df2.set_index('id'),on='id2',how='inner')
        cand_df = cand_df[['id1','joinKey1','id2','joinKey2']]
        return cand_df


    def verification(self, cand_df, threshold):
        #Function to calculate Jaccard Index
        def JCIndex(k1,k2):
            inter = len(set(k1).intersection(k2))
            unio = len(set(k1).union(k2))
            jaccard_val = abs(inter/unio)
            return jaccard_val

        #Creating a new column named jaccard and call JCIndex to obtain similarity index
        cand_df['jaccard'] = np.vectorize(JCIndex)(cand_df["joinKey1"],cand_df["joinKey2"])
        # Filter with respect to threshold
        result_df = cand_df[cand_df['jaccard'] >= threshold]
        return result_df


    def evaluate(self, result, ground_truth):
        T_set = set(map(tuple,result)) & set(map(tuple,ground_truth))
        T_val = len(list(map(list, T_set)))
        R_val = len(list(map(list,result)))
        A_val = len(list(map(list,ground_truth)))

        precision = abs(T_val)/abs(R_val)
        recall = abs(T_val)/abs(A_val)

        F_measure = (2 * precision * recall) / (precision + recall)
        return precision,recall,F_measure


    def jaccard_join(self, cols1, cols2, threshold):

        new_df1 = self.preprocess_df(self.df1, cols1)
        new_df2 = self.preprocess_df(self.df2, cols2)
        print ("Before filtering: %d pairs in total" %(self.df1.shape[0] *self.df2.shape[0]))

        cand_df = self.filtering(new_df1, new_df2)
        print ("After Filtering: %d pairs left" %(cand_df.shape[0]))

        result_df = self.verification(cand_df, threshold)
        print ("After Verification: %d similar pairs" %(result_df.shape[0]))

        return result_df


if __name__ == "__main__":
    dataset = input("Please pass either amazongoogle-sample or amazongoogle as input below to run this program. Other inputs are not accepted. Ensure both .csv datasets and code are present in same folder \n")
    proceed = 'n'
    if dataset == 'amazongoogle-sample':
        set1 = "Amazon_sample.csv"
        set2 = "Google_sample.csv"
        truth = "Amazon_Google_perfectMapping_sample.csv"
        proceed = 'y'
    elif dataset == 'amazongoogle':
        set1 = "Amazon.csv"
        set2 = "Google.csv"
        truth = "Amazon_Google_perfectMapping.csv"
        proceed = 'y'


    if proceed == 'y':
        #er = SimilarityJoin("Amazon_sample.csv", "Google_sample.csv")
        er = SimilarityJoin(set1, set2)
        amazon_cols = ["title", "manufacturer"]
        google_cols = ["name", "manufacturer"]
        result_df = er.jaccard_join(amazon_cols, google_cols, 0.5)

        result = result_df[['id1', 'id2']].values.tolist()
        #ground_truth = pd.read_csv("Amazon_Google_perfectMapping_sample.csv").values.tolist()
        ground_truth = pd.read_csv(truth).values.tolist()
        print ("(precision, recall, fmeasure) = ", er.evaluate(result, ground_truth))
    else:
        print("Input argument provided is invalid or empty.Please rerun the program and provide correct input(amazongoogle-sample or amazongoogle)")
