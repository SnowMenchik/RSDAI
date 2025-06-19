# Импорт библиотек
import cv2  # Работа с видео и изображениями
import numpy as np  # Математические операции с массивами
from ultralytics import YOLO  # Детекция объектов
import time  # Работа с временными интервалами

# Загрузка предобученной модели YOLO (ваша кастомная модель)
model = YOLO('best.pt')

def draw_info_table(frame, sign, speed, light):
    # Получаем размеры оригинального кадра
    height, width = frame.shape[:2]  # shape возвращает (высота, ширина, каналы)
    
    # Создаем белую панель для информации
    table_height = 100  # Высота информационной панели
    table = np.full((table_height, width, 3), 255, dtype=np.uint8)  # 255 - белый цвет
    
    # Рисуем черную рамку вокруг таблицы
    cv2.rectangle(table, (0, 0), (width-1, table_height-1), (0, 0, 0), 5)
    
    # Заголовки колонок
    headers = ["SIGN", "SPEED", "LIGHT"]  # Названия колонок
    col_width = width // 3  # Ширина каждой колонки
    
    # Рисуем вертикальные разделители
    for i in range(1, 3):  # Для 2 разделителей (между 3 колонками)
        cv2.line(table, (i*col_width, 0), (i*col_width, table_height), (0, 0, 0), 3)
    
    # Настройки шрифта для заголовков
    font = cv2.FONT_HERSHEY_SIMPLEX  # Стандартный шрифт OpenCV
    scale = 0.5  # Масштаб текста
    thickness = 1  # Толщина линий букв
    
    # Рисуем заголовки колонок
    for i, header in enumerate(headers):  # i - индекс (0,1,2), header - значение
        x = i*col_width + 80  # Позиция X для текста (сдвиг на 80 пикселей)
        cv2.putText(table, header, (x, 25), font, scale, (0, 0, 0), thickness)
    
    # Настройки для значений
    scaler = 1  # Масштаб значений больше чем у заголовков
    values = [sign, speed, light]  # Список значений для отображения
    
    # Рисуем значения в колонках
    for i, value in enumerate(values):
        x = i*col_width + 20  # Позиция X (сдвиг на 20 пикселей)
        # Красный цвет если значение отсутствует (равно "-")
        color = (0, 0, 0) if value != '-' else (0, 0, 255)
        cv2.putText(table, str(value), (x, 70), font, scaler, color, thickness)
    
    # Соединяем оригинальный кадр с таблицей по вертикали
    return np.vstack([frame, table])  # vstack - вертикальная конкатенация

def main():
    # Инициализация видеопотока с веб-камеры (индекс 0 - дефолтная камера)
    cap = cv2.VideoCapture(0)
    
    # Временные метки для обновления данных каждые 0.5 секунды
    next_update_time = time.time() + 0.5
    
    # Инициализация переменных состояния
    speed_value = "-"  # Текущее ограничение скорости
    light_color = "-"  # Текущий цвет светофора
    sign_flag = "-"    # Текущий дорожный знак

    # Основной цикл обработки видео
    while cap.isOpened():  # Проверка подключения камеры
        # Чтение кадра
        ret, frame = cap.read()  # ret - флаг успешности, frame - изображение
        if not ret:  # Если кадр не получен
            break     # Выход из цикла

        # Детекция объектов с помощью YOLO
        results = model(frame, verbose=False)  # verbose=False - отключение логов
        
        # Визуализация результатов детекции (автоматическая отрисовка bbox)
        annotated_frame = results[0].plot()

        # Обновление данных каждые 0.5 секунды
        if time.time() >= next_update_time:
            # Временные переменные для текущего кадра
            current_speed = "-"
            current_light = "-"
            current_sign = "-"

            # Перебор всех обнаруженных объектов
            for box in results[0].boxes:
                # Получаем ID класса
                class_id = int(box.cls[0])  # cls - тензор с классами
                # Получаем название класса из словаря names
                class_name = model.names[class_id]
                # Уверенность предсказания (приводим к float)
                conf = box.conf.item()
                
                # Фильтр по уверенности (пропускаем если <0.5)
                if conf <= 0.5:
                    continue  # Пропустить текущую итерацию
                
                # Обработка знака "Stop"
                if 'Stop' in class_name:
                    current_sign = "Stop"  # Устанавливаем флаг
                
                # Обработка ограничения скорости
                elif 'Speed Limit' in class_name:
                    # Разбиваем название класса на части (например: "Speed Limit 60")
                    parts = class_name.split()
                    if len(parts) > 2:  # Проверяем наличие числовой части
                        # Проверяем что третья часть - число
                        if parts[2].isdigit():
                            current_speed = parts[2]  # Сохраняем значение
                
                # Обработка светофора
                elif class_name in ['Red Light', 'Green Light']:
                    # Определяем цвет по названию класса
                    if 'Red' in class_name:
                        current_light = 'Red'
                    else:
                        current_light = 'Green'

            # Обновление глобальных переменных только при обнаружении
            if current_speed != "-":
                speed_value = current_speed  # Сохраняем новое значение скорости
            if current_light != "-":
                light_color = current_light  # Обновляем цвет светофора
            if current_sign != "-": 
                sign_flag = current_sign  # Фиксируем обнаружение знака

            # Устанавливаем время следующего обновления
            next_update_time = time.time() + 0.5

        # Создаем комбинированный кадр с информацией
        combined_frame = draw_info_table(annotated_frame, sign_flag, speed_value, light_color)
        
        # Отображаем результат в окне
        cv2.imshow('Road Sign Detection', combined_frame)

        # Условия выхода: нажатие 'q' или закрытие окна
        if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty('Road Sign Detection', cv2.WND_PROP_VISIBLE) < 1:
            break

    # Освобождаем ресурсы
    cap.release()  # Закрываем видеопоток
    cv2.destroyAllWindows()  # Уничтожаем все окна OpenCV

if __name__ == "__main__":
    main()  # Запуск программы при непосредственном выполнении файла