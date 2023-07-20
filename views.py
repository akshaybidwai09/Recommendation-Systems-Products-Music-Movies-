from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
import json
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import jsonlines
import pandas as pd
import re
import math

import os
from django.conf import settings

# os.environ.setdefault("DJANGO_SETTINGS_MODULE", "myproject.settings")
# settings.configure()


class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyArrayEncoder, self).default(obj)


def get_data_frame():
    data = []
    with jsonlines.open(
        "D:\AkshayFiles\Data Preprocessing in AI ML\FACE DETECTIONS\datasetcleaned.ldjson"
    ) as reader:
        for obj in reader:
            data.append(obj)

    return pd.DataFrame(data)


def convert_to_ndarray(text2_num_vector_string):
    text2_num_vector_string = text2_num_vector_string[1:-1]
    text2_num_vector_list = re.findall(
        r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?", text2_num_vector_string
    )
    text2_num_vector_float = [float(value) for value in text2_num_vector_list]
    text2_num_vector = np.array(text2_num_vector_float)
    return text2_num_vector


def find_similar_products(request, product_id, num_similar):
    df5 = get_data_frame()
    product_row = df5[df5["uniq_id"] == product_id]
    if product_row.empty:
        return ValueError("Product Not found")
    text1_num_vector = convert_to_ndarray(product_row["Vectorized_attributes"].iloc[0])

    similarity_scores_map = {}
    for idx, row in df5.iterrows():
        if row["uniq_id"] != product_id:
            text2_num_vector = convert_to_ndarray(row["Vectorized_attributes"])
            similarity_score = cosine_similarity([text2_num_vector], [text1_num_vector])
            # similarity_score = math.trunc(similarity_score[0][0] * 1000) / 1000
            similarity_scores_map[row["uniq_id"]] = (
                similarity_score,
                row["product_name"],
                row["rating"],
                row["sales_price"],
            )

    similarity_scores_map = sorted(
        similarity_scores_map.items(), key=lambda x: (x[1][0], x[1][2]), reverse=True
    )
    similar_products = {}
    index = 0
    for key, val in similarity_scores_map:
        if num_similar == 0:
            break
        products = {
            "Unique_id": key,
            "Similarity_score": val[0],
            "product_Name": val[1],
            "rating": val[2],
            "sales_price": val[3],
        }
        similar_products[index] = products
        index += 1
        num_similar -= 1

    return JsonResponse(similar_products, encoder=NumpyArrayEncoder, safe=False)


def convert_to_ndarray(text2_num_vector_string):
    text2_num_vector_string = text2_num_vector_string[1:-1]
    text2_num_vector_list = re.findall(
        r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?", text2_num_vector_string
    )
    text2_num_vector_float = [float(value) for value in text2_num_vector_list]
    text2_num_vector = np.array(text2_num_vector_float)
    return text2_num_vector


def find_similar_productsV1(request, product_id, num_similar):
    df5 = get_data_frame()

    product_row = df5[df5["uniq_id"] == product_id]
    if product_row.empty:
        return ValueError("Product Not found")
    text1_num_vector = convert_to_ndarray(product_row["Vectorized_attributes"].iloc[0])

    text1_num_vector = text1_num_vector.reshape(1, -1)

    data_vectors = np.array(
        [
            convert_to_ndarray(vector_string)
            for vector_string in df5["Vectorized_attributes"]
        ]
    )

    data_vectors_id = df5["uniq_id"].values
    ratings = df5["rating"].values
    data_vectors = data_vectors.reshape(data_vectors.shape[0], -1)

    similarity_scores = cosine_similarity(text1_num_vector, data_vectors)

    map_final = {}
    for i in range(similarity_scores.shape[1]):
        if data_vectors_id[i] != product_row["uniq_id"].iloc[0]:
            map_final[data_vectors_id[i]] = tuple([similarity_scores[0][i], ratings[i]])

    map_final = sorted(
        map_final.items(), key=lambda x: (x[1][0], x[1][1]), reverse=True
    )

    answer = {}
    index = 0

    for key, val in map_final:
        if num_similar == 0:
            break
        products = {"uniq_id": key, "similarity_score": val[0], "rating": val[1]}
        answer[index] = products
        index += 1
        num_similar -= 1

    return JsonResponse(answer, encoder=NumpyArrayEncoder, safe=False)

    #             #         ANNOY          #              #


from annoy import AnnoyIndex


def getAnnoyInstance(df5):
    annoy_index_loaded = AnnoyIndex(
        len(convert_to_ndarray(df5["Vectorized_attributes"].iloc[0])), metric="angular"
    )
    annoy_index_loaded.load("annoy_index_dj.ann")
    return annoy_index_loaded


def find_similar_products_ann(request, product_id, num_similar):
    df5 = get_data_frame()
    print(df5.shape)
    annoy_instance = getAnnoyInstance(df5)
    product_row = df5[df5["uniq_id"] == product_id]
    if product_row.empty:
        return ValueError("No Product Found")
    query_vector = convert_to_ndarray(product_row["Vectorized_attributes"].iloc[0])
    similar_vectors = annoy_instance.get_nns_by_vector(query_vector, num_similar + 1)
    final_products = []
    final = {}
    indx = 0

    for idx in similar_vectors:
        row = df5[df5["annoy_index"] == idx].iloc[0]
        if row["uniq_id"] != product_id:
            final_products.append(row["uniq_id"])
            products = {
                "product_id": row["uniq_id"],
                "product_name": row["product_name"],
                "rating": row["rating"],
            }
            final[indx] = products
            indx += 1
    return JsonResponse(final, encoder=NumpyArrayEncoder, safe=False)


import annoy
from annoy import AnnoyIndex


def build_annoy_network(df):
    vector_size = len(convert_to_ndarray(df["Vectorized_attributes"].iloc[0]))
    annoy_index = annoy.AnnoyIndex(vector_size, metric="angular")
    df["annoy_index"] = range(len(df))
    for _, row in df.iterrows():
        vector = convert_to_ndarray(row["Vectorized_attributes"])
        annoy_index.add_item(row["annoy_index"], vector)
    num_trees = 100
    annoy_index.build(num_trees)

    return annoy_index


# annoy_index = build_annoy_network(get_data_frame())
# annoy_index.save("annoy_index_dj.ann")


# print(find_similar_products_ann({}, "37f854d6b0fa64e237fe01d97944f51d", 4))
