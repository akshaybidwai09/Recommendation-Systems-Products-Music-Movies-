# Recommendation-Systems-(Amazon Products)
This Project is build on the Django Microservice, so the idea is to fetch most similar amazon products based on a certain product purchase by a user. This approach is often used in Amazon shopping cart where similar products are recommended based on the current products on the cart. 

Dataset Used - https://www.kaggle.com/datasets/promptcloud/amazon-fashion-products-2020

# Details
I have used Approximate Nearest Neighbors (Annoy Network) and (Similarity Batch Search)
Similarity search is based on a few attributes that helped to evaluate Cosine Similarity (Scikit-learn module) with other products. (Dataset- Amazon Data (Kaggle) with
30K records).
This Design is built using linear batch search with average response time of ~600 ms and later developed using Annoy network with response time

