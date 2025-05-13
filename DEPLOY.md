# Публикация TradeFusion Simulator для общего доступа

## Введение

Этот документ содержит инструкции по публикации приложения TradeFusion Simulator для общего доступа, чтобы другие пользователи могли использовать симулятор через интернет без необходимости локальной установки.

## Варианты публикации

### 1. Streamlit Cloud (рекомендуется)

Samый простой и бесплатный способ опубликовать Streamlit-приложение:

1. **Создайте репозиторий на GitHub**:
   - Создайте новый репозиторий на GitHub
   - Загрузите все файлы проекта в репозиторий (включая `streamlit_app.py`, `trade_fusion_simulation.py` и `requirements.txt`)

2. **Опубликуйте на Streamlit Cloud**:
   - Перейдите на [share.streamlit.io](https://share.streamlit.io/)
   - Войдите с помощью аккаунта GitHub
   - Нажмите "New app"
   - Выберите ваш репозиторий с TradeFusion
   - В поле "Main file path" укажите `streamlit_app.py`
   - Нажмите "Deploy"

3. **Поделитесь ссылкой**:
   - После успешного деплоя вы получите публичный URL (например, `https://username-tradefusion-streamlit_app-py.streamlit.app`)
   - Этой ссылкой можно поделиться с любым пользователем

### 2. Heroku

Для публикации на Heroku:

1. **Подготовьте файлы для деплоя**:
   - Создайте файл `Procfile` в корневой директории проекта со следующим содержимым:
     ```
     web: streamlit run streamlit_app.py --server.port=$PORT
     ```
   - Убедитесь, что файл `requirements.txt` содержит все необходимые зависимости

2. **Создайте приложение на Heroku**:
   - Создайте аккаунт на [Heroku](https://www.heroku.com/)
   - Установите [Heroku CLI](https://devcenter.heroku.com/articles/heroku-cli)
   - Выполните следующие команды в терминале:
     ```bash
     heroku login
     heroku create tradefusion-app  # Замените на уникальное имя
     git init
     git add .
     git commit -m "Initial commit"
     git push heroku master
     ```

3. **Откройте приложение**:
   - После успешного деплоя выполните команду `heroku open`
   - Или перейдите по URL вида `https://tradefusion-app.herokuapp.com`

### 3. Pythonanywhere

Для публикации на Pythonanywhere (подходит для бесплатного хостинга):

1. **Создайте аккаунт**:
   - Зарегистрируйтесь на [PythonAnywhere](https://www.pythonanywhere.com/)

2. **Загрузите файлы**:
   - Перейдите в раздел "Files"
   - Создайте новую директорию для проекта
   - Загрузите все файлы проекта

3. **Настройте виртуальное окружение**:
   - Откройте консоль Bash
   - Создайте виртуальное окружение:
     ```bash
     mkvirtualenv tradefusion --python=python3.9
     pip install -r requirements.txt
     ```

4. **Настройте веб-приложение**:
   - Перейдите в раздел "Web"
   - Добавьте новое веб-приложение
   - Выберите "Manual configuration" и Python 3.9
   - Настройте путь к виртуальному окружению: `/home/yourusername/.virtualenvs/tradefusion`
   - Добавьте WSGI файл со следующим содержимым:
     ```python
     import sys
     import os
     
     path = '/home/yourusername/tradefusion'
     if path not in sys.path:
         sys.path.append(path)
     
     from streamlit.web.bootstrap import run
     
     def application(environ, start_response):
         os.environ['STREAMLIT_SERVER_PORT'] = environ['SERVER_PORT']
         run()
     ```

## Дополнительные рекомендации

### Настройка для больших объемов данных

Если ваша симуляция работает с большими объемами данных:

- На Streamlit Cloud: увеличьте лимиты памяти в файле `.streamlit/config.toml`
- На Heroku: используйте платные планы с большим объемом памяти

### Безопасность

Если вы планируете использовать приложение для реальных данных:

- Добавьте аутентификацию пользователей с помощью `streamlit-authenticator`
- Используйте HTTPS для защиты передаваемых данных

### Оптимизация производительности

Для улучшения производительности приложения:

- Используйте кэширование с помощью декораторов `@st.cache_data` и `@st.cache_resource`
- Оптимизируйте загрузку и обработку данных
- Разделите приложение на несколько страниц с помощью `st.session_state`

## Устранение проблем

### Проблемы с зависимостями

Если возникают проблемы с установкой зависимостей:

- Укажите точные версии библиотек в `requirements.txt`
- Проверьте совместимость версий Python и библиотек

### Ошибки при деплое

Если приложение не запускается после деплоя:

- Проверьте логи приложения на платформе деплоя
- Убедитесь, что все необходимые файлы загружены
- Проверьте пути к файлам в коде