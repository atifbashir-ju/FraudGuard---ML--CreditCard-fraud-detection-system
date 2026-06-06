from django.urls import path
from src.dashboard import views

urlpatterns = [
    path('',                        views.index,               name='index'),
    path('dashboard/predict/',      views.predict,             name='predict'),
    path('dashboard/batch/',        views.batch_upload,        name='batch_upload'),
    path('dashboard/history/',      views.transaction_history, name='history'),
    path('dashboard/model-info/',   views.model_info,          name='model_info'),
    path('dashboard/export/',       views.export_csv,          name='export_csv'),
]
