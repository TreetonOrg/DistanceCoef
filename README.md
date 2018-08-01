# DistanceCoef

### Выводы
* Текущий результат: https://github.com/TreetonOrg/DistanceCoef/blob/master/results/regressor_2.csv, linear_regression: 0.080 MSE, 0.705 корреляция
* Лучший результат:  https://github.com/TreetonOrg/DistanceCoef/blob/master/results/regressor_1.csv, xgboost: 0.053 MSE, 0.799 корреляция.
* Для внедрения нужна реализация или обёртка бустинга под Java (например http://xgboost.readthedocs.io/en/latest/jvm/java_intro.html). Непонятно, насколько это нужно, возможный выигрыш: MSE 0.080 -> 0.053, Spearman 0.705 -> 0.799
* Линейная регрессия всех выходах Тритона (https://github.com/TreetonOrg/DistanceCoef/blob/master/results/regressor_1.csv) даёт лучший коэффициент коррелляции Спирмена, но иногда выбрасывает сильно за пределы от 0 до 1, и MSE ужасный.

### Файлы и директории
* data - папка с входными tsv файлами.
* run_treeton.sh - скрипт, запускающий разборы папок.
* train.py - самый главный скрипт. Выводит коэффициенты модели.
* train.sh - скрипт с параметрами по умолчанию.
* requirements.txt - набор требуемых пакетов.

### Механизм работы
Модель линейной регрессии такая же, как и в коде Тритона.

Общая схема:
* Берём данные.
* Прогоняем через поэтический разборщик Тритона.
* Получаем нужные коэффициенты по обеим строкам.
* Получаем обучающую выборку.
* Обучаемся, коэффициенты переносим в настройки Тритона.

Алгоритмы для задачи регрессии:
* LINEAR_REGRESSION - линейная регрессия.
* LINEAR_SVR - SVM.
* DECISION_TREE - дерево решений.
* RANDOM_FOREST
* XGBOOST

Признаки:
* TREETON_BASE: сырые коэффициенты из Тритона.
* TREETON_AGG: агрегированные коэффициенты из Тритона.
* MANUAL: ручные признаки - разница в длине строк, в количестве гласных.
* CHAR_GRAMS: символьные n-граммы.

Нормировка для регрессии: 1 - ((x - 1) / 4), 5->0, 1->1

Собираемые метрики для регрессии:
* Корреляция Спирмена: ранговая корреляция, чем ближе к 1, тем лучше.
* MSE, mean squared error: средний квадрат разности оценок. Чем ближе к 0, тем лучше.

### Результаты
Папка results, файлики regressor_*.csv. Цифра - набор фич:
* 1 - TREETON_BASE
* 2 - TREETON_AGG
* 3 - TREETON_AGG + TREETON_BASE
* 4 - MANUAL
* 5 - MANUAL + TREETON_BASE
* 6 - MANUAL + TREETON_AGG
* 7 - MANUAL + TREETON_AGG + TREETON_BASE

### Использование
Установка зависимостей:
```
sudo pip3 install -r requirements.txt
```

Запуск: 
```
sh train.sh <путь к папке distrib c verseProcessingTool.sh>
```

Если verseProcessingTool.sh нет, то нужно из verseProcessingTool.bat его переделать

Последняя строка вывода - коэффициенты регрессии, переносим их в настройки Тритона.
Коэффициенты переносятся в том же порядке, без потери знаков!
Формат в настройках:
```
regressionCoefficients=<coef1>;<coef2>;<coef3>;<coef4>;<coef5>;<coef6>;<coef7>
```

Текущие коэффициенты:
-0.12170645900901182;-0.16122737322232428;-0.11879022371283338;-0.06921142952268064;-0.20374728842474554; 0.0;0.5612788899501985
