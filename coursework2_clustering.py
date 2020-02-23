import pandas as pd
import numpy as np
from random import uniform, randint, choice
from statistics import mean
import matplotlib.pyplot as plt
import matplotlib.cm as cm


iris_class = ['Iris Setosa', 'Iris Versicolour', 'Iris Virginica']
iris_attributes = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width', 'Type']
wine_attributes = ['Class','Alcohol', 'Malic Acid', 'Ash', 'Alcalinity of Ash', 'Magnesium',
              'Total Phenols', 'Flavanoids', 'Nonflavanoid Phenols', 'Proanthocyanins',
              'Color Intensity', ' Hue', 'OD280/OD315 of Diluted Wines', 'Proline']


def get_data(data='iris'):
    '''
    Gets the iris or wine data sets.
    param iris: A string indicating what data to get. Default data set to get is the iris data set.
    return: Return iris data set is iris parameter is true else returns the win
    '''
    if data == 'iris':
        file_to_read = 'iris.txt'
        names = iris_attributes
    elif data == 'wine':
        file_to_read = 'wine.txt'
        names = wine_attributes
    elif data == 'cluster':
        file_to_read = 'cluster_validation_data.txt'
        names = range(1,3)
    else:
        #Return false to indicate the data was not valid
        return False
    data = pd.read_csv(file_to_read,
                     sep=',',
                     names=names)
    return data



#Data constants
wine_data = get_data('wine')[wine_attributes[1:]].values
iris_data = get_data('iris')[ iris_attributes[:-1] ].values
cluster_data = get_data('cluster').values



def standardise_data(data):
    '''
    Used to standardise the data
    param data: The data to standardise
    return: The standardised data
    '''
    data_mean = np.mean(data, axis = 0)
    data_standard_deviation = np.std(data, axis = 0)
    
    return (data - data_mean) / data_standard_deviation



#Standardise data constants
std_wine_data = standardise_data( np.copy(wine_data)  )
std_iris_data = standardise_data( np.copy(iris_data) )
std_cluster_data = standardise_data( np.copy(cluster_data) )




def get_euclidean_distance(point1, point2):
    '''
    Calculate the Euclidean distance between two points
    param centroid: The first point
    param instance: The second point
    return: The Euclidean distance between the two points
    '''
    #Calculate difference between each value then square them
    difference = np.array(point1) - np.array(point2)
    square_difference = [x**2 for x in difference]
    
    return np.sqrt(np.sum(square_difference))



def which_centroid(centroids, x):
    '''
    Determines which centroid a point belong to
    param centroids: A list of centroids
    param x: The point to test. This is one instance of the data
    return: The index of the corresponding centroid in the centroids list
    '''
    
    #keep track of the minimum distance and the what centroid this distance corresponds to
    min_distance_to_centroid = get_euclidean_distance(centroids[0], x)
    current_centroid = 0
    
    for centroid in range( 1, len(centroids) ):
        distance_to_centroid = get_euclidean_distance(centroids[centroid], x)
        
        if distance_to_centroid < min_distance_to_centroid:
            min_distance_to_centroid = distance_to_centroid
            current_centroid = centroid
    
    return np.array(current_centroid)


def calculate_centroid(cluster):
    '''
    Calculates the centroid of a cluster
    param cluster: The cluster
    return: The centroid of the cluster
    '''
    
    #average the points in the cluster to get the centroid
    point_sum = np.sum(cluster, axis = 0)
    number_of_points = len(cluster)
    return np.array(point_sum/number_of_points)



def kmeans(x, k, max_itr = 100):
    '''
    An algorithm which implements the k-means cluster algorithm
    param x: The data to be clustered as a pandas DataFrame
    param k: The number of clusters
    max_itr: The maximum number of iterations
    return: The cluster memeber labels for each elememt in the data x
    '''
    
    #create the initial centroids 
    centroids = [ choice(x) for centroid in range(k) ]
    membership_labels = []
    
    for i in range(max_itr):
        clusters = [ [] for cluster in range(k) ]
        
        membership_labels = []
        
        #assign each instance to a cluster
        for instance in x:
            which_cluster = which_centroid(centroids, instance) 
            clusters[ which_cluster ].append(instance)
            #keep a record of which cluster the instance belongs to
            membership_labels.append( which_cluster )
        
        new_centroids = []
        
        #update the centroids
        for i in range( len(clusters) ):
            
            if clusters[i] == []:
                new_centroids.append( centroids[i] )
            else:
                new_centroids.append( calculate_centroid( clusters[i] ) )
        
        #check if the algorithm has converged
        if np.array_equal( centroids, new_centroids ):
            break
        else:
            centroids = new_centroids
    
    return np.array(membership_labels)


def calculate_average_distance(centroid, cluster):
    '''
    Calculate the average difference of all points in a cluster to its centroid
    param centroid: The centroid of the cluster
    param cluster: The cluster 
    return: The average distancce of the every data point of the cluster to its centroid
    '''
    
    total_distance = 0
    
    for instance in cluster:
        total_distance += get_euclidean_distance(centroid, instance)
        
    return total_distance/len(cluster)  

def davies_bouldin(x, cluster_labels):
    '''
    Calculate the Davies Bouldin index for a cluster
    param x: The data points
    param cluster_labels: A list indicate which cluster a point in x belongs to
    return: The Davies Bouldin index of the solution
    '''
    
    
    number_of_clusters = max(cluster_labels) + 1
    
    #recreate the clusters using the labels
    clusters = [ [] for x in range(number_of_clusters) ]    
    for i in range(x.shape[0]):
        clusters[cluster_labels[i]].append(x[i])    
    
    #calculate the centroid of each cluster
    centroids = [ calculate_centroid(x) for x in clusters ]
    
    #calculate the average distance of all points in a cluster to its center, for each cluster
    average_centroid_distance = [ calculate_average_distance(centroid, instance) 
                                 for centroid, instance in zip(centroids, clusters) ]
    
    
    sum_Dij = 0
    #calculate the sum Dij
    for i in range( number_of_clusters ):      
        max_Dij = 0
        
        #calculate the max Dij
        for j in range( number_of_clusters ):            
            if i == j:
                continue
                
            #calculate di + dj
            numerator = average_centroid_distance[i] + average_centroid_distance[j]
            
            #calculate dij
            denominator = get_euclidean_distance( centroids[i], centroids[j] )
            
            current_Dij = numerator/denominator
            
            if max_Dij < current_Dij:
                max_Dij = current_Dij
                
        sum_Dij += max_Dij
    
    return sum_Dij / number_of_clusters



def single_silhouette_score(clusters, current_point, current_cluster):
    '''
    Calculate the silhouette score for a single data point
    param clusters: The data points in their corresponding cluster
    param current_point: The data point to calculate a silhouette score for
    return: The silhouette score of a single data point
    '''
    #Handle condition when |C| is 1
    if len(clusters[current_cluster]) == 1:
        return 0
    
    #Calculate Ai
    
    distance_ai = 0
    for instance in clusters[current_cluster]:
        distance_ai += get_euclidean_distance(instance, current_point)
    
    cluster_size_ai = len(clusters[current_cluster]) - 1
    
    ai = distance_ai / cluster_size_ai
    
    #Calculate bi
    all_bi = []
    for i in range( len(clusters) ):
        #skip the cluster the current point is in
        if current_cluster == i:
            continue
            
        distance_bi = 0
        cluster_size_bi = len( clusters[i] )
        for instance in clusters[i]:
            distance_bi += get_euclidean_distance(instance, current_point)
            
        all_bi.append( distance_bi / cluster_size_bi )
    
    bi = min( all_bi )
    
    #calculate the silhouette score
    if ai > bi:
        return ( bi - ai ) / ai
    else:
        return ( bi - ai ) / bi


def silhouette_scores(x, cluster_labels): 
    '''
    Calculates the silhouette score for every data point 
    param x: The data points
    param cluster_labels: The membership cluster label for each data point in x
    return: The silhouette score for each data point in x.
    '''
    number_of_clusters = max(cluster_labels) + 1
    #Use cluster_label to recreate the cluster
    clusters = [ [] for x in range(number_of_clusters) ]    
    for i in range(x.shape[0]):
        clusters[cluster_labels[i]].append(x[i])   
    
    scores = []
    #calculate silhouette score for each point
    for i in range(x.shape[0]):
        current_point = x[i]
        current_cluster = cluster_labels[i]
        
        scores.append(single_silhouette_score(clusters, current_point, current_cluster))
    
    return np.array(scores)

if __name__ == '__main__':
    print( kmeans( std_wine_data, 3 ) )