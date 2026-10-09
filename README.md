# pytrendclust-examples

Воспроизводимые примеры и эксперименты к статье о библиотеке
[pytrendclust](https://pypi.org/project/pytrendclust/)
(исходный код: <https://github.com/89605502155/pytrendclust>).

Блокнот `examples.ipynb` строит рисунки 3–5 и таблицы 1–2 статьи:

* рис. 3: зашумлённая парабола, два кластера (спад и рост);
* рис. 4: знакопеременный ряд «забор», один кластер плато;
* рис. 5: синусоида из пяти периодов, 11 чередующихся кластеров;
* табл. 1: доля верно восстановленных разбиений при разных штрафах;
* табл. 2: ошибка прогноза потребления памяти при разных обучающих выборках.

Все ряды генерируются с фиксированными зёрнами генератора, поэтому
результаты совпадают при каждом запуске. Рисунки сохраняются в `figures/`
в форматах PNG и TIFF с разрешением 600 dpi.

## Запуск

```bash
uv sync
uv run jupyter nbconvert --to notebook --execute --inplace examples.ipynb
```

Полный прогон занимает около получаса, основное время уходит на таблицы 1 и 2.
Ячейки блокнота генерируются скриптом `build_notebook.py`.

## Reproducible examples for the pytrendclust paper

The notebook reproduces Figures 3–5 and Tables 1–2 of the paper. Install
[uv](https://docs.astral.sh/uv/) and run the two commands above.
