# Stereo Middlebury 3D Reconstruction

Программная реализация для статьи:

**«Анализ эффективности методов стереозрения для восстановления 3D-облака точек по карте диспаритета на датасете Middlebury»**.

Проект строит карты диспаритета по стереопарам Middlebury, сравнивает методы `StereoBM` и `StereoSGBM`, рассчитывает ошибки относительно эталонной карты диспаритета и восстанавливает 3D-облако точек.

## Структура проекта

```text
stereo_middlebury_project/
  data/
    raw/                    # сюда скачивается архив Middlebury
    middlebury_2006/         # сюда распаковывается датасет
  outputs/                   # результаты экспериментов
  src/
    config.py
    download_dataset.py
    main.py
    metrics.py
    pipeline.py
    point_cloud.py
    stereo_methods.py
    utils.py
    visualize.py
  requirements.txt
  run_experiment.bat
```

## Установка

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Быстрый запуск

Скачать датасет:

```bash
python -m src.main --download
```

Запустить эксперимент по всем сценам:

```bash
python -m src.main
```

Запустить только по нескольким сценам:

```bash
python -m src.main --scenes Aloe Cloth1 Flowerpots
```

## Что создаётся в outputs

Для каждой сцены создаётся папка:

```text
outputs/Aloe/
  input_left.png
  input_right.png
  epipolar_lines.png
  gt_disparity.png
  bm_disparity.png
  bm_error.png
  bm_depth.png
  bm_point_cloud.ply
  bm_point_cloud_preview.png
  sgbm_disparity.png
  sgbm_error.png
  sgbm_depth.png
  sgbm_point_cloud.ply
  sgbm_point_cloud_preview.png
```

Также создаются общие файлы:

```text
outputs/summary_metrics.csv
outputs/metrics_mae.png
outputs/metrics_rmse.png
outputs/metrics_bad2.png
outputs/metrics_coverage.png
```

## Как использовать результаты в статье

В статье можно вставить:

1. `epipolar_lines.png` — иллюстрация стереопары и горизонтальных эпиполярных линий.
2. `gt_disparity.png` — эталонная карта диспаритета Middlebury.
3. `bm_disparity.png` и `sgbm_disparity.png` — сравнение двух методов.
4. `bm_error.png` и `sgbm_error.png` — карты ошибок относительно ground truth.
5. `bm_point_cloud_preview.png` и `sgbm_point_cloud_preview.png` — визуализация 3D-облака точек.
6. `summary_metrics.csv` и графики `metrics_*.png` — количественное сравнение.

## Важно про масштаб Middlebury 2006 ThirdSize

В проекте используется версия `ThirdSize/zip-2views/ALL-2views.zip`. Для неё карты диспаритета из датасета делятся на 3, так как исходные disparity map в уменьшенной версии требуют такого масштабирования. Для восстановления глубины используется приближённая формула:

```text
Z = f * B / d
```

где `f = 3740 / 3` пикселей, `B = 160` мм, `d` — диспаритет в пикселях уменьшенного изображения.

Для качественного сравнения методов и иллюстрации 3D-восстановления этого достаточно.
