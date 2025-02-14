import os
import json
import requests
import openpyxl
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.http import JsonResponse, FileResponse
from .forms import UploadFileForm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Корень проекта
JSON_PATH = os.path.join(BASE_DIR, 'file_get', 'templates', 'request.json')  # Путь до request.json

def upload_file(request):
    if request.method == 'POST' and request.FILES['file']:
        uploaded_file = request.FILES['file']
        fs = FileSystemStorage()
        filename = fs.save(uploaded_file.name, uploaded_file)
        file_path = fs.path(filename)

        # Выводим имя файла в локальный терминал
        print(f"Выбран файл: {uploaded_file.name}")  # Это будет выводиться в локальный терминал
        print(f"Выбран файл: {filename}")
        # 📌 ✅ Читаем JSON из файла
        with open(JSON_PATH, 'r', encoding='utf-8') as json_file:
            json_data = json.load(json_file)

        # 📌 ✅ Ищем URL для "Analyse file Copy"
        analyse_file_url = None
        for item in json_data.get("item", []):
            if item.get("name") == "Analyse file Copy":
                analyse_file_url = item["request"]["url"]["raw"]
                break

        if not analyse_file_url:
            return JsonResponse({"error": "API URL not found in JSON"}, status=500)

        # 📌 ✅ Отправляем файл пользователя
        with open(file_path, 'rb') as f:
            files = {'file': (filename, f, 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')}
            response = requests.post(analyse_file_url, files=files)

        if response.status_code == 200:
            # 📌 ✅ Получаем `XLSX`-файл от API
            excel_data = response.content
            output_filename = "processed_" + filename
            output_path = fs.path(output_filename)

            with open(output_path, 'wb') as output_file:
                output_file.write(excel_data)

            # 📌 ✅ Парсим `XLSX` в HTML-таблицу
            table_html = parse_excel_to_html(output_path)

            # 📌 ✅ Передаём путь к файлу в шаблон
            return render(request, 'result.html', {
                'table_html': table_html,
                'file_url': fs.url(output_filename)  # Путь для скачивания
            })

        return JsonResponse({"error": "Ошибка запроса к API"}, status=response.status_code)

    else:
        form = UploadFileForm()
    return render(request, 'upload.html', {'form': form})

def parse_excel_to_html(file_path):
    """Функция парсинга `XLSX` в HTML"""
    workbook = openpyxl.load_workbook(file_path)
    sheet = workbook.active

    table_html = "<table border='1' cellpadding='5'>"
    for row in sheet.iter_rows(values_only=True):
        table_html += "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
    table_html += "</table>"

    return table_html
