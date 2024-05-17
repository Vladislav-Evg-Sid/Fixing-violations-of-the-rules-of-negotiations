from flask import render_template
from config import app

@app.route("/")
def index() -> str:
    """Функция позволяет отрендерить главную страницу веб-сервиса.

    Returns:
        str: отрендеренная главная веб-страница.
    """
    
    return render_template(
        "index.html",
        page_title="Сводка",
    )
