"""Untitled4.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1u2mergsZ8Pnu1Nju8I5Tlnw4IAy1_5dY

## Анализ аудио
"""

# !pip install vosk
# !pip install pydub
# !pip install ruptures

import os
from vosk import Model, KaldiRecognizer, SetLogLevel
from pydub import AudioSegment
from pydub.effects import normalize, low_pass_filter, high_pass_filter
from pydub.silence import split_on_silence
import json
import librosa
import numpy as np
import matplotlib.pyplot as plt
import ruptures as rpt
import math

SetLogLevel(0)

def analysis(audio_path):


    # Устанавливаем частоту дискретизации и количество каналов
    FRAME_RATE = 16000
    CHANNELS = 1

    # Загрузка модели vosk-model-ru-0.42
    model = Model("E:/git/RZD/Fixing-violations-of-the-rules-of-negotiations/web/model")

    # Создание объекта распознавателя речи
    rec = KaldiRecognizer(model, FRAME_RATE)
    rec.SetWords(True)

    # Загрузка аудиофайла
    # audio_path = 'НУЖНЫЙ_ФАЙЛ.mp3'
    mp3 = AudioSegment.from_mp3(audio_path)

    # Чтение аудиофайла с помощью librosa
    y, sr = librosa.load(audio_path, sr=FRAME_RATE, mono=True)

    # Деление аудиофайла на промежутки с голосом
    intervals = librosa.effects.split(y, top_db=30, frame_length=2048, hop_length=512)

    # Фильтрация шумов и улучшение качества голоса
    audio_chunks = split_on_silence(mp3, min_silence_len=1000, silence_thresh=-30, keep_silence=True)
    filtered_audio = AudioSegment.silent(duration=500)

    # Вычисление средней громкости для голосовых сегментов
    total_voice_dBFS = 0
    voice_chunk_count = 0

    for chunk in audio_chunks:
        if len(chunk) < 1000:  # Пропускаем слишком короткие сегменты
            continue
        chunk = normalize(chunk)  # Нормализация громкости
        chunk = low_pass_filter(chunk, cutoff=50)  # Фильтрация низких частот
        chunk = high_pass_filter(chunk, cutoff=400)  # Фильтрация высоких частот
        chunk_dBFS = chunk.dBFS  # Получаем уровень громкости в децибелах

        # Проверка уровня громкости для голосовых сегментов
        if chunk_dBFS > -30:  # Порог тишины
            total_voice_dBFS += chunk_dBFS
            voice_chunk_count += 1

    if voice_chunk_count > 0:
        average_voice_dBFS = total_voice_dBFS / voice_chunk_count
    else:
        average_voice_dBFS = -30

    # Усиление голосовых сегментов на основе средней громкости
    for chunk in audio_chunks:
        if len(chunk) < 1000:  # Пропускаем слишком короткие сегменты
            continue
        chunk_dBFS = chunk.dBFS

        # Усиление голосовых сегментов на основе средней громкости
        if (chunk_dBFS < average_voice_dBFS) and (chunk_dBFS > average_voice_dBFS - 7):
            chunk = chunk + (average_voice_dBFS - chunk_dBFS) + 15  # Усиление
        filtered_audio += chunk

    audio_chunks = split_on_silence(filtered_audio, min_silence_len=1000, silence_thresh=-30, keep_silence=True)
    filtered_audio = AudioSegment.silent(duration=0)

    # Определите допустимый интервал между интервалами
    max_interval_gap = 400  # в миллисекундах

    # Список для хранения групп интервалов
    interval_groups = []

    # Создание первой группы
    current_group = [intervals[0]]

    # Перебор оставшихся интервалов
    for interval in intervals[1:]:
        # Вычисление промежутка времени между текущим и предыдущим интервалом
        time_gap = librosa.samples_to_time(interval[0] - current_group[-1][-1], sr=FRAME_RATE) * 1000

        # Если промежуток времени между интервалами меньше допустимого значения, добавляем интервал в текущую группу
        if time_gap <= max_interval_gap:
            current_group.append(interval)
        else:
            # Если промежуток больше допустимого значения, заканчиваем текущую группу и создаем новую
            interval_groups.append(current_group)
            current_group = [interval]

    # Добавление последней группы
    interval_groups.append(current_group)

    # Объединение интервалов в каждой группе
    for group in interval_groups:
        # Получение начального и конечного времени группы
        start_sample = group[0][0]
        end_sample = group[-1][1]

        # Создание аудиофрагмента для текущей группы
        group_audio = y[start_sample:end_sample]

        # Преобразование аудио в формат AudioSegment
        group_audio = (group_audio * 32767).astype(np.int16)
        group_audio = AudioSegment(group_audio.tobytes(), frame_rate=FRAME_RATE, sample_width=2, channels=1)

        # Фильтрация шумов и улучшение качества голоса

        # Добавление обработанного промежутка к итоговому аудио
        filtered_audio += group_audio

    filtered_audio = filtered_audio + 2

    # Установка частоты дискретизации и количества каналов
    filtered_audio = filtered_audio.set_channels(CHANNELS)
    filtered_audio = filtered_audio.set_frame_rate(FRAME_RATE)

    # Сохранение обработанного аудио
    filtered_audio_path = "filtered_audio.wav"
    filtered_audio.export(filtered_audio_path, format="wav")

    # Загрузка аудиофайла
    y, sr = librosa.load(filtered_audio_path)

    # Создание спектрограммы
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, fmin=50, fmax=500)

    # Применение скользящего среднего для сглаживания данных
    smoothed_spectrogram = librosa.util.normalize(librosa.power_to_db(librosa.feature.stack_memory(spectrogram, n_steps=10, mode='edge'), ref=np.max))

    # Вычисление средней энергии по частотам для каждого временного окна
    mean_energy = np.mean(smoothed_spectrogram, axis=0)

    # Преобразование временных данных в массив numpy
    X = mean_energy.reshape(-1, 1)

    # Инициализация алгоритма сегментации
    algo = rpt.Pelt(model="l1").fit(X)
    result = algo.predict(pen=1.4)
    # display
    rpt.display(X,result)
    plt.show()

    # Визуализация спектрограммы
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spectrogram, sr=sr, x_axis='time', vmax=80)  # Указываем желаемый диапазон значений дБ

    # Отмечаем временные моменты на спектрограмме
    for point in result[:-1]:
        plt.axvline(x=point * librosa.frames_to_time(1), color='r', linestyle='--')

    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-Spectrogram with Change Points')
    plt.show()

    # Получение длины аудиозаписи в секундах
    audio_length = librosa.get_duration(y=y, sr=sr)

    # Перевод значений концов сегментов из частей в тайминг аудиозаписи
    segment_timings = [(endpoint / result[-1]) * audio_length for endpoint in result]

    # Разбиение аудиозаписи на сегменты
    segments = []
    for i in range(len(result)):
        start_time = 0 if i == 0 else segment_timings[i - 1]  # Начальное время сегмента
        end_time = segment_timings[i]  # Конечное время сегмента
        segment = y[int(start_time * sr):int(end_time * sr)]
        segments.append(segment)

    print(segments)

    # Создадим список для хранения времен начала и конца каждого сегмента
    segment_timings = []

    # Переменная для хранения времени начала текущего сегмента
    start_time = 0

    # Рассчитаем время начала и конца каждого сегмента
    for segment in segments:
        # Длительность текущего сегмента
        segment_duration = len(segment) / sr

        # Время конца текущего сегмента
        end_time = start_time + segment_duration

        # Добавляем время начала и конца текущего сегмента в список
        segment_timings.append((start_time, end_time))

        # Обновляем время начала для следующего сегмента
        start_time = end_time

    # В переменной segment_timings теперь хранятся времена начала и конца каждого сегмента
    print(segment_timings)

    # for segment in segments:
    #     # Распознавание речи и преобразование вывода в json
    #     rec.AcceptWaveform(segment.raw_data)
    #     result = rec.Result()
    #     text = json.loads(result)["text"]
    #     print(result)

    # import librosa

    # # Функция для вычисления средней энергии по частотам (в диапазоне от 50 до 500 Гц) для каждого коридора
    # def compute_corridor_energy(spectrogram, result, sr):
    #     corridor_energy = []
    #     for i in range(len(result) - 1):
    #         start_frame = result[i]
    #         end_frame = result[i + 1]
    #         # Определяем частоты, соответствующие диапазону от 50 до 500 Гц
    #         freq_min = librosa.core.note_to_hz('C1')  # Минимальная частота, соответствующая ноте C1 (примерно 32 Гц)
    #         freq_max = librosa.core.note_to_hz('C4')  # Максимальная частота, соответствующая ноте C4 (примерно 500 Гц)
    #         # Находим соответствующие индексы частот в спектрограмме
    #         idx_min = librosa.core.fft_frequencies(sr=sr)
    #         idx_max = librosa.core.fft_frequencies(sr=sr)
    #         idx_min = np.argmax(idx_min >= freq_min)
    #         idx_max = np.argmax(idx_max >= freq_max)
    #         # Вычисляем среднюю энергию по частотам внутри текущего коридора и диапазона частот
    #         energy = np.mean(spectrogram[idx_min:idx_max, start_frame:end_frame], axis=(0, 1))
    #         corridor_energy.append(energy)
    #     return np.array(corridor_energy)

    # # Вычисляем среднюю энергию по частотам (в диапазоне от 50 до 500 Гц) для каждого коридора
    # corridor_energy = compute_corridor_energy(smoothed_spectrogram, result, sr)

    # # Выводим размерность массива с средней энергией для каждого коридора
    # print("Размерность массива с средней энергией для каждого коридора:", corridor_energy)

    # def compute_random_energy(spectrogram, result, num_samples=10):
    #     random_energy = []
    #     for i in range(len(result) - 1):
    #         start_frame = result[i]
    #         end_frame = result[i + 1]
    #         # Генерируем случайные индексы для текущего коридора
    #         random_indices = np.random.randint(start_frame, end_frame, size=(num_samples, 2))
    #         # Вычисляем среднюю энергию для каждого случайного участка внутри коридора
    #         energies = []
    #         for indices in random_indices:
    #             # Проверяем, что участок не пустой
    #             if indices[1] > indices[0]:
    #                 energy = np.mean(spectrogram[:, indices[0]:indices[1]])
    #                 energies.append(energy)
    #         # Если список энергий не пуст, добавляем среднее значение
    #         if energies:
    #             random_energy.append(np.mean(energies))
    #         else:
    #             # Если список энергий пуст, добавляем NaN
    #             random_energy.append(np.nan)
    #     return np.array(random_energy)

    # # Вычисляем средние рандомные значения энергии для каждого коридора
    # random_energy = compute_random_energy(smoothed_spectrogram, result, num_samples=10)

    # # Выводим полученные значения
    # print("Средние рандомные значения энергии для каждого коридора:", random_energy)


    # print("Массив result (change points):", result)



    # !pip install pyAudioAnalysis hmmlearn eyed3

    data_path = 'data.txt'

    # Функция для применения распознавания речи к сегменту и записи результатов в файл
    def recognize_segment(segment, start_time, end_time, rec):
        # Получение аудио сегмента
        segment_audio = segment.raw_data

        # Распознавание речи и преобразование вывода в JSON
        rec.AcceptWaveform(segment_audio)
        result = rec.Result()
        text = json.loads(result)["text"]

        # Запись результатов в файл, добавляя информацию о времени сегмента
        with open(data_path, 'a') as f:  # 'a' для добавления в файл
            f.write(f"[{start_time}:{end_time}]\n")
            f.write(text + "\n\n")  # Добавляем пустую строку для разделения результатов сегментов

    if os.path.exists(data_path) and os.path.getsize(data_path) > 0:
        # Файл существует и не пустой, очищаем его содержимое
        open(data_path, 'w').close()

    # Загрузка аудиофайла
    audio = AudioSegment.from_wav(filtered_audio_path)

    # Проходим по каждому сегменту и его временным таймингам
    for (start_time, end_time) in segment_timings:
        # Вырезаем сегмент из аудиофайла
        segment = audio[math.floor(start_time * 100)*10:math.floor(end_time * 100)*10]  # Переводим секунды в миллисекунды

        # Применяем распознавание речи к сегменту и записываем результаты в файл
        recognize_segment(segment, math.floor(start_time*100)/100, math.floor(end_time*100)/100, rec)

    rec.AcceptWaveform(filtered_audio.raw_data)
    result_load = rec.Result()
    text = json.loads(result_load)["text"]
    

    # with open(data_path, 'a') as f:
    #     f.write(f"[0.00:{round(segment_timings[-1][1],2)}]\n")
    #     f.write(text)
    
    return(text)