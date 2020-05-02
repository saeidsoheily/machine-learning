# __author__ = 'Saeid SOHILY-KHAH'
# Machine learning algorithms: K-means Clustering
if (!"useful" %in% rownames(installed.packages())){
  install.packages('useful');
}
library(useful) # a set of useful functions such as plotting the results of K-means clustering

if (!"cluster" %in% rownames(installed.packages())){
  install.packages('cluster');
}
library(cluster) # methods for cluster analysis

if (!"NbClust" %in% rownames(installed.packages())){
  install.packages('cluster', dependencies = TRUE);
}
library(NbClust) # package for determining the best number of clusters


# Function: data preprocessing
preprocessingData <- function(data){
  # Remove any records that have NAs
  data = na.omit(data)
  
  # Split data to features and labal column
  features <- data[, which(names(data) != "Label")]     # features
  scaled_features <- as.matrix(scale(features))         # scaled features
  
  label <- data[, which(names(data) == "Label")]        # label
  return(list(scaled_features, label))
}


# Function: elbow method for finding the optimal number of clusters
plotElbow <- function(data, max_n_clusters) {
  # Compute and plot Within-cluster Sum of Squared errors (WSS) 
  wss <- Inf
  for (n_clusters in 1:max_n_clusters){
    wss[n_clusters] <- sum(kmeans(data, centers=n_clusters)$withinss)
  }
  plot(1:max_n_clusters, 
       wss, type="b", 
       xlab="Number of Clusters", 
       ylab="Within-cluster Sum of Squared errors (WSS) ", 
       main = "ELBOW METHOD")
  axis(1, at = seq(1, max_n_clusters, by = 1)) # define the axis labels
}


# Function: silhouette method for finding the optimal number of clusters
plotSilhouette <- function(data, max_n_clusters) {
  # Compute and plot Average Silhouette Value
  msv <- Inf
  for (n_clusters in 2:max_n_clusters){
    silhouette_value <- silhouette(kmeans(data, centers=n_clusters)$cluster, dist(data))
    msv[n_clusters] <-  mean(silhouette_value[, 3])
  }
  plot(1:max_n_clusters, 
       msv, type="b", 
       xlab="Number of Clusters", 
       ylab="Average Silhouette Value", 
       main = "SILHOUETTE METHOD")
  axis(1, at = seq(2, max_n_clusters, by = 1)) # define the axis labels
}


# Function: NbClust
nbClust <- function(data, max_n_clusters){
  # provides 30 indices for determining the number of clusters and proposes to user the best clustering scheme 
  nb <- NbClust(data, 
                diss=NULL, 
                distance = "euclidean",  
                max.nc=max_n_clusters, 
                method = "kmeans", 
                index = "all", 
                alphaBeale = 0.1)
  hist(nb$Best.nc[1,], 
       breaks = max(na.omit(nb$Best.nc[1,])), 
       col = "pink", 
       xlab = "Number of clusters", 
       ylab = "Frequency")
  axis(1, at = seq(1, max_n_clusters, by = 1)) # define the axis labels
}


# ------------------------------------------------ MAIN ----------------------------------------------------
# Load data
raw_data_url <- 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data' 
column_names <- c('Label', 'Alcohol', 'Malic-acid', 'Ash', 'Alcalinity-of-ash', 'Magnesium', 'Total-phenols', 'Flavanoids', 
                  'Nonflavanoid-phenols', 'Proanthocyanin', 'Color-intensity', 'Hue', 'OD280-OD315-of-diluted.wines', 'Proline')
raw_data <- read.csv(raw_data_url, header=FALSE, sep=',', col.names=column_names) 

# Preprocessing data
returned_list <- preprocessingData(raw_data)
features <- returned_list[[1]]   # features
label <- returned_list[[2]]      # label

# Analyze the data 
summary(as.data.frame(unclass(features)))

# Plot elbow method
plotElbow(features, max_n_clusters = 10)

# Plot silhouettte method
plotSilhouette(features, max_n_clusters = 10)

# Plot histogram of NbClust
nbClust(features, max_n_clusters = 10)   

# Run k-means clustering 
# Set the number of clusters (for kmeans clustering)
best_n_clusters <- length(unique(label)) # or using elbow, silhouettte, or nbClust,.. methods for non-labelled data
kmeans_clusters <- kmeans(features, centers=best_n_clusters, nstart = 25)
kmeans_clusters # summarize the result

# Plot k-means clustering results
plot(kmeans_clusters, data=features)

# Confusion matrix
cm <- table(label, kmeans_clusters$cluster) 
cm # confusion matrix

# Plot confusion matrix
plot(cm, main="CONFUSION MATRIX", xlab="Actual Label", ylab="Cluster")