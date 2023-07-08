![screenshot7](https://github.com/ocaoimh/data_bootcamp_assignments-/blob/main/Lab%20%7C%20Unsupervised%20learning%20intro/data/Screenshot%202023-07-08%20at%2010.48.06.png)


# Lab | Unsupervised learning intro

#### Description

We had to create a song recommendation application. Users are asked to enter a song title and the recommender will suggest a song that is similar. 

To do this, I used an unsupervised learning technique called k-means to cluster songs by their audio features. Audio features were added to each song title using the Spotify database and accessed via the Spotify API.


#### Clusters
In order to find the best clusters, we had to test different ways of grouping and analyse the features of each cluster. 



<img src="https://github.com/ocaoimh/data_bootcamp_assignments-/blob/main/Lab%20%7C%20Unsupervised%20learning%20intro/data/Screenshot%202023-07-08%20at%2010.54.04.png" width=70% height=70%>

_Quick glance at the clusters _

<img src="https://github.com/ocaoimh/data_bootcamp_assignments-/blob/main/Lab%20%7C%20Unsupervised%20learning%20intro/data/Screenshot%202023-07-08%20at%2010.53.58.png" width=70% height=70%>

_Looking for the best elbow - 4 looks to to be good_

<img src="https://github.com/ocaoimh/data_bootcamp_assignments-/blob/main/Lab%20%7C%20Unsupervised%20learning%20intro/data/Screenshot%202023-07-08%20at%2010.53.52.png" width=70% height=70%>

_Silhouette score - it's a toss up between 3 and 5 I'll go for 3 as the red line overreach and the relative widths are more homogeneous_

