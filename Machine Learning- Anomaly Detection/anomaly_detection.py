import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter

#pd.set_option('display.max_colwidth', None)


class AnomalyDetection():

    def scaleNum(self, df, indices):
        #Remove NaN values from dataframe
        df = df.dropna()

        #Flatten array into columns
        feature1 = pd.DataFrame(df['Features'].values.tolist(), index= df.index)

        #Loop through each dimensions
        for ft in indices:
            #Tag each feature into a new dataframe
            feature = feature1[ft]
            feature = feature.astype(float)

            # Perform the scaling function.
            feature = (feature - feature.mean()) / feature.std()
            feature = feature.replace(np.NaN, 0)

            #Reassign the scaled values back to dataframe
            feature1[ft] = feature

        #Convert the dataframe columns back to list
        df['Features'] = feature1.values.tolist()
        dfnew = df[['Features']]
        dfnew["id"] = df.index
        dfnew = dfnew[['id', 'Features']].set_index('id')
        return dfnew

    def cat2Num(self, df, indices):
        # Two lists are created for category 1 and category 2
        cat_1 = []
        cat_2 = []
        cat_cnt1 = 0
        cat_cnt2 = 0
        # Separate the numerical features out
        num_features = df["features"].map(lambda x: x[2:])

        #Loop through indices to encode both the columns
        for ind in indices:
            feature = df["features"].map(lambda x: x[ind])

            #Obtain unique values for each feature
            feature_uni = feature.unique()

            #Create a dictionary with index for list of features
            feature_cnt = len(feature_uni)

            # For two categorical features we are globalizing the unique count to be checked in scale2num function
            if ind == indices[0]:
                cat_cnt1 = feature_cnt
            else:
                cat_cnt2 = feature_cnt

            # Creating a new dictionary with unique values to be checked with dataframe
            feature_dict = {}
            for i in range(0,feature_cnt):
                feature_dict[i] = feature_uni[i]

            #Check through individual rows to check if the value is present in unique list
            for ids in feature:
                feature_list = []
                if ids in feature_uni:
                    # Obtain the index key of matching value
                    keys = [key for key, value in feature_dict.items() if value == ids][0]
                    # Loop through the count of unique elements. Append 1 for the matching index key and others as zeros.
                    for j in range(0,feature_cnt):
                        if j == keys:
                            feature_list.append(1)
                        else:
                            feature_list.append(0)

                    #Since we have two columns to be encoded, two lists are maintained to stored encodings of each column
                    if ind == indices[0]:
                        cat_1.append(feature_list)
                    else:
                        cat_2.append(feature_list)

        # Combine all the features after one hot encoding to form the transformed feature list
        list_f = [l1 + l2 + l3 for l1, l2, l3 in zip(cat_1, cat_2,num_features)]

        #Create a new dataframe with transformed columns
        dfnew = pd.Series(list_f)
        dfnew = dfnew.to_frame('Features')
        dfnew["id"] = df.index
        dfnew = dfnew[['id', 'Features']].set_index('id')

        #Creating the total count of one hot encoded features to be used in scale2num function.
        global cat_cnt
        cat_cnt = cat_cnt1 + cat_cnt2

        return dfnew


    def detect(self, df, k, t):

        #Calculation of score
        def scorecal(cnt):
            N_x = cnt
            score = (N_max - N_x) / (N_max - N_min)
            return score

        # Converting the dataframe which is having features as object to a numpy array to be used for scikit learn
        X = pd.DataFrame(df["Features"].values.tolist())

        #Apply K Means algorithm
        kmeans = KMeans(n_clusters=k,random_state=0).fit(X)
        #frequency = Counter(labels).values()

        labels_df = pd.DataFrame(kmeans.labels_,columns =['label'])
        #print(labels_df.groupby(['label']).size().reset_index(name='counts'))

        df['count'] = labels_df.groupby('label')['label'].transform('count')
        # Calculate minimum and maximum frequency
        N_max = df['count'].max()
        N_min = df['count'].min()

        # Calculation of the score using map
        df["score"] = df["count"].map(scorecal)
        df = df.drop(columns=['count'])

        #Obtain those anomalies whose score is greater than threshold
        df = df[df["score"] >= t]
        return df

if __name__ == "__main__":

    # Function defined to perform anomaly detection part
    def Process(scalex=0,scaley=0,clusters=2,threshold=0.9):
        ad = AnomalyDetection()

        df1 = ad.cat2Num(df, [0, 1])
        print(df1)

        # Uncomment this line if you need to pass a range of consecutive dimensions
        # df2 = ad.scaleNum(df1, [*range(scalex,scaley,1)])

        df2 = ad.scaleNum(df1, [scalex])
        print(df2)

        df3 = ad.detect(df2, clusters, threshold)
        print(df3)

    # Starting of the process
    dataset = input(
        "Please pass either toy or sample or full as input below to run this program. Other inputs are not accepted \n")
    proceed = 'n'
    if dataset == 'toy':
        # Loads toy dataset
        data = [(0, ["http", "udt", 4]), \
                (1, ["http", "udf", 5]), \
                (2, ["http", "tcp", 5]), \
                (3, ["ftp", "icmp", 1]), \
                (4, ["http", "tcp", 4])]
        df = pd.DataFrame(data=data, columns=["id", "features"]).set_index('id')

        # Parameters to be passed for scaling and K Means
        scalex = 6
        scaley = 7
        clusters = 2
        threshold = 0.9

        #Call below to process anomaly detection
        Process(scalex,scaley,clusters,threshold)

    elif dataset == 'sample':
        #Loads sample dataset
        df = pd.read_csv('logs-features-sample.csv',converters={"features": lambda x: x.strip("[]").replace("'","").split(", ")}).set_index('id')

        # Parameters to be passed for scaling and K Means
        scalex = [13,14]
        clusters = 8
        threshold = 0.97

        # Pass this paramter if you need a range. If you are using a range then scale x variable should be an integer and not list
        scaley = 0
        #Call below to process anomaly detection
        Process(scalex,scaley,clusters,threshold)

    elif dataset == 'full':
        df = pd.read_csv('logs-features.csv',low_memory=False,converters={"rawFeatures": lambda x: x.strip("[]").replace("'","").split(", ")}).set_index('id')
        df.columns = ['features']
        # Parameters to be passed for scaling and K Means
        scalex = [15]
        clusters = 8
        threshold = 0.97
        # Pass this paramter if you need a range. If you are using a range then scale x variable should be an integer and not list
        scaley = 0
        #Call below to process anomaly detection
        Process(scalex,scaley,clusters,threshold)
    else:
        print("Input argument provided is invalid or empty.Please rerun the program and provide correct input(toy or sample or full)")

