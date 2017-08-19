# DistanceCoef

### Файлы и директории:
* ratings.csv - файл с обучающими данными, 5 колонок: requestId, songId, shift, size, rhyme. Разделитель ";". 
⋅⋅* requestId - Id запроса из папки requests.
⋅⋅* songId - Id песни из папки songs.
⋅⋅* shift - сдвиг, по которому мы ищем отрывок в песне.
⋅⋅* size - собственно то, что приближаем. Оценки от 1 до 5. При обучении меняет диапазон на [0, 1].
⋅⋅* rhyme - при обучении не используется.
* requests - папка с изнчальными запросами
* songs - папка с песнями
* song_texts - служебная папка, в которую кладутся вырезанные отрывки песен и их разборы Тритоном
* request_texts - служебная папка, в которую кладутся запросы и их разборы Тритоном.
* run_treeton.sh - скрипт, запускающий разборы папок song_texts и request_texts

### Механизм работы:
Модель линейной регрессии такая же, как и в коде Тритона. 
Берём данные, вырезаем нужные фрагменты, прогоняем через поэтический разборщик Тритона, 
получаем нужные коэффициенты по каждой строке обоих фрагментов, преобразуем их, получаем обучающую выборку, 
обучаемся, коэффициенты переносим в настройки Тритона.

### Использование
Установка зависимостей:
```
sudo pip3 install -r requirements.txt
```

Запуск: 
```
python3 train.py <путь к папке c verseProcessingToolDistance.sh> <1, если с результатами кросс-валидации, 0 иначе>
```

Последняя строка вывода - коэффициенты регрессии, переносим их в настройки Тритона.
Коэффициенты переносятся в том же порядке!
Формат в настройках:
```
regressionCoefficients=<coef1>;<coef2>;<coef3>;<coef4>;<coef5>
```

Если нужно добавить данных - нужно менять ratings.csv и презапустить обучение.