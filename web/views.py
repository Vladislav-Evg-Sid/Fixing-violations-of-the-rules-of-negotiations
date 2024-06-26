from flask import render_template, redirect, url_for, request
from text_analysis import text_analysis
from config import app
from stt_module import analysis

dialogues = [["29к_874 КВ - 02.05.2024 01_08_44.mp3", "ДНЦ: Машинист поезда 22-го на приближении к станции Погромное (или Красногвардеец- 2 Погромное). ДНЦ Бахтинова\nТЧМ: 22-ой машинист Карабин, слушает Вас\nДНЦ: До станции Сорочинская проедьте. По Тоцкой по первому пути будет ехать. ДНЦ Бахтинова\nТЧМ: Понятно. С Тоцкой по первому пути до Сорочинск максимально допустимой следуем, машинист Карабин", 0],
              ["30к_872 КВ - 02.05.2024 08_40_27.mp3", "ДНЦ: Машинист поезда 123 на 3-ем пути станции Заливное.\nТЧМ: Машинист поезда 123 на 3-м пути станции Заливное, Глошев. Слушаю вас.\nДНЦ: Машинист поезда 123 Подтянули вплотную к сигналу Н-3?\nТЧМ: Да, вплотную встали. Перекрывайте сигнал Н-3. Машинист Глошев.\nДНЦ: Понятно, перекрываю сигнал Н-3\nТЧМ: Прибытие 8:34\nДНЦ: Понятно. Прибытие 8:34", 0],
              ["63к_824 КВ - 02.05.2024 04_21_45.mp3", "ДНЦ: 3576 у входного станции Есаульская, <ответьте поездному>\nТЧМ: 3576 на перегоне Ишалино Есаульская, у входного станции Есаульская, машинист Васильев, слушаю вас\nДНЦ: Машинист до Шагла <разгоняемся, проходы оперативнее делаем,> на скрещение с Орланом <будем> по станции Шагол\nТЧМ: Понятно, скрещение с Орланом по станции Шагол 3576 машинист Васильев", 1, 
               ["Диспетчер обратился к машинисту не по форме", "Диспетчер дал не существующую команду", "Диспетчер не подтвердил действие машиниста"]],
              ["71к_855 КВ - 02.05.2024 01_34_17.mp3", "ДНЦ: 3004 <на первом начальная>\nТЧМ: Слушаю 3004-ый на первом Начальном машинист\nДНЦ: С первого Ч-1 открыт, отправляейтесь ДНЦ Чиж\nТЧМ: Понятно, выходной с первого Ч-1 два желтых верхний мигающий отправляемся 3004 машинист\nДНЦ: <На втором А>", 1,
               ["Диспетчер обратился к машинисту не по форме", "Диспетчер ведет посторонний разговор"]]]

def analyze_lines(line2):
    # Это типо диалог от Димы.
    line = [
        'Машинист поезда №2120 на 5 - м пути станции К.',
        'Слушаю Вас, машинист поезда №2120 Иванов',
        'Приказ №1 время 1: 30(один час, тридцать минут). Разрешаю машинисту отправиться с 5 - го пути по четному главному пути при запрещающем показании выходного светофора Ч5 и следовать до выхода на перегон со скоростью 20 км / час, а далее руководствоваться сигналами локомотивного светофора. ДСП Петрова.',
        'Понятно. Приказ №1: 30(один час, тридцать минут).Разрешаете поезду №2120 отправиться с 5 - го пути по четному главному пути при запрещающем показании выходного светофора Ч5 и следовать до выхода на перегон со скоростью 20 км / час, а далее руководствоваться сигналами локомотивного светофора. Машинист поезда №2120 Иванов.',
        'Верно, выполняйте'
    ]
    # Анализ
    analysis = text_analysis.text_analysis()
    return analysis.analyze(line2)

def words_in_brackets(input):
    arr = []
    while True:
        if input.find("<") == -1:
            break
        start = input.find("<")
        end = input.find(">")
        arr.append(input[:start])
        arr.append(input[start + 1:end])
        input = input[end+1:]
    arr.append(input)
    return arr 

@app.route("/")
def analyze() -> str:
    """Функция позволяет отрендерить главную страницу веб-сервиса.

    Returns:
        str: отрендеренная главная веб-страница.
    """
    return render_template(
        "index.html",
        dialogues=dialogues,
        words_in_brackets=words_in_brackets,
        page_title="Анализатор служебных переговоров",
    )

@app.route("/clear")
def clear():
    global dialogues
    dialogues = []
    return redirect(url_for("analyze"))

@app.route("/upload", methods=["POST"])
def upload():
    global dialogues
    files = request.files['formFileMultiple']
    for file in files:
        dialogues.append(analyze_lines(analysis(file)))
    return redirect(url_for("analyze"))