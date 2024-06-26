from text_analysis import text_analysis


def main() -> str:
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
    return analysis.analyze(line)


if __name__ == '__main__':
    main()