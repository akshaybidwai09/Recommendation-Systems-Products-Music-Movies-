from django.urls import path

from . import views

urlpatterns = [
    path("<str:product_id>/<int:num_similar>/", views.find_similar_products, name="api"),
    path("batch/<str:product_id>/<int:num_similar>/",views.find_similar_productsV1),
    path("ann/<str:product_id>/<int:num_similar>/",views.find_similar_products_ann)
    ]
