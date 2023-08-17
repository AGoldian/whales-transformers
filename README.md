# **Application for identification of large marine mammals**

Репозиторий содержит полную реализацию веб-приложения, предназначенного для идентификации крупных морских млекопитающих, таких как киты и дельфины. Это инновационное приложение позволяет отслеживать миграцию и популяцию этих важных морских существ семейства.

Основные функции:
+ Идентификация морских млекопитающих: Приложение использует передовые методы компьютерного зрения и машинного обучения для распознавания индивидуальных особей морских млекопитающих на основе их уникальных характеристик, таких как окраска, пятна и форма.
+ Реализован интуитивно понятный веб-интерфейс, который позволяет пользователям легко взаимодействовать с приложением. Загрузка изображений морских млекопитающих, просмотр результатов и настройка параметров анализа доступны через удобный пользовательский интерфейс.

===
### **Установка и запуск**

Clone the repo and change to the project root directory:
```
git clone https://github.com/AGoldian/whales-transformers.git
cd whales
```

Install requirements:
```
pip install -r requirements.txt
```

And run:
```
streamlit run streamplit_app.py
```

## **Используемое решение:**

Технологии для решения отбирались по двум критериям - это автономность работы решения и наличие библиотек открытого доступа. 
После чего мы провели сравнительный анализ, выбрав модели оптимальные по соотношению точности к требуемой производительности: VisionTransformers (8 multihead Attention block)

## **Технические детали:**

+ Язык программирования: Python
+ Веб-фреймворк: Streamlit
+ Библиотеки машинного обучения: PyTorch, OpenCV, Albumentations
+ Система контроля версий: Git