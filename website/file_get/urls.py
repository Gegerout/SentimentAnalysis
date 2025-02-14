from django.conf import settings
from django.conf.urls.static import static
from django.urls import path
from .views import upload_file

urlpatterns = [
    path('', upload_file, name='upload_file'),
]

# 📌 ✅ Даем Django доступ к загруженным файлам
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

