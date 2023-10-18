# Recommendation Systems for Amazon Products
This project is a Django microservice aimed at providing recommendations for similar Amazon products based on a user's product purchase. The approach used here is commonly employed in online shopping platforms like Amazon, where customers are presented with related products based on their current selections.

# Dataset
The project utilizes the Amazon Fashion Products dataset from Kaggle, which can be accessed (https://www.kaggle.com/datasets/promptcloud/amazon-fashion-products-2020). This dataset contains a vast collection of Amazon fashion products and serves as the foundation for generating product recommendations (30k Records for this project).

# Approach
To provide efficient and accurate recommendations, two techniques have been employed in this project:

Approximate Nearest Neighbors (Annoy Network): Annoy is a library that facilitates fast nearest neighbor searches. In this system, Annoy is used to efficiently find the most similar products to a given user's product.

Similarity Batch Search: For similarity search, the system focuses on specific attributes that are vital in evaluating the cosine similarity between products. This is achieved using the Scikit-learn module in Python. The dataset, consisting of approximately 30,000 records from the Amazon Data on Kaggle, is employed to calculate and generate meaningful product recommendations.

# Performance
The initial implementation employed a linear batch search, resulting in an average response time of approximately 600 milliseconds. However, as the dataset and the number of products grew, the response time became a concern.

To address this, the system was later improved using the Annoy network, which significantly optimized the response time for generating recommendations.

# How it works
The Django microservice receives a user's product as input.
The system then identifies the relevant attributes of the product to compute its cosine similarity with other products in the dataset.
If the batch search approach is utilized, the system performs a linear search for similar products.
If the Annoy network is utilized, the system efficiently finds the nearest neighbors, greatly reducing response times.
The microservice returns a list of recommended Amazon products based on the similarity scores calculated using either the batch search or the Annoy network.
Conclusion
By leveraging the power of Django, Scikit-learn, and Annoy, this project provides an effective and scalable recommendation system for Amazon products. The Annoy network ensures that response times remain low even as the dataset grows, offering users a seamless shopping experience with relevant product suggestions based on their preferences and purchases.
