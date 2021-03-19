# Signal-Processing - решение задачи с семантическим разрывом

## Выбранные объекты:
- Объект А - воротца;
- Объект Б - пластина.

## Требования к объектам и окружению:
Даны воротца и пластина. 
- Пластина влезает в нишу ворот в каких-то положениях, в каких-то не влезает, это зависит от угла ее поворота и того, на какой из граней она лежит.
- Цвета ворот и пластины контрастны друг другу.
- Ниша ворот и пластина прямоугольной формы.
- Задник на фотографии белый, а поверхность-пол - белая клетчатая тетрадная бумага.

## Требования к фотографиям:
- Фотографии сделаны на камеру с разрешением не менее 9.5МП.
- Освещение искусственное, без засвеченных областей и теней по интенсивности сопоставимых с самими объектами.
- Объекты полностью попадают в кард и в фокусе.
- Угол съемки  - на пролет ворот, всегда постоянен с точностью 2-3 градуса (условия лабораторные и камера фиксирована).

## Задача:

Требуется определить, влезет ли пластина в нишу при параллельном переносе(бинарная классификация - YES, если влезет, NO, если не влезет). В качестве дополнения может быть рассмотрена задача подсчета угла поворота требуемого для прохождения пластины в нишу с точностью до 1 градуса.


## Датасет:

Ознакомиться с предварительными образцами выборки можно по ссылке - https://drive.google.com/drive/folders/1XL2db2UVi6GiDcnMneswp40IV3YnX9xg?usp=sharing.

## Алгоритм решения
- сглаживание фильтром Гаусса
- бинаризация Оцу
- морфологические операции
- поиск контуров с помощью cv.findContours
- построение ограничивающего прямоугольника
- вычисление его пропорций
- сравнение пропорции с порогом и принятие рещения

## Идея
Для рассматриваемой в задаче пластинки существует некоторое пороговое положение, при котором она еще пролезает в ворота. В терминах ограничивающего прямоугольника это положение с учетом постоянства угла съемки задается вне зависимости от близости пластинки к камере некоторым соотношением сторон ограничивающего пластику прямоугольника(bounding box). Высчитываем это пороговое значение соотношения сторон(width/height) по специально снятой фотографии этого краевого случая. Для поступающей на вход фотографии высчитываем то же соотношение и, если оно больше - ответ "NO", если меньше - ответ "YES".

<img width="937" alt="idea" src="https://user-images.githubusercontent.com/39533142/111794561-6643d600-88d7-11eb-99e5-c43745f1951c.png">

## Полученные результаты и над чем еще надо работать
- точность достигается пока не очень большая - 0.64.
- это так из-за ошибок при нахождении ограничивающих прямоугольников.
- далее для улучшения работы модели можно подумать над другими способами нахождения bounding boxes. 

## Запуск программы
Код на питоне - main.py можно запустить через командную строку так: **python main.py -i <path_to_image>**
