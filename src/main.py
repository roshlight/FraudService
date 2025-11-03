import pandas as pd
from pathlib import Path
from .preprocess import preprocess
from .inference import load_model_and_features, predict
from .postprocess import save_submission, save_importances, save_density_plot
from .config import INPUT_DIR, OUTPUT_DIR, ARTIFACTS_DIR

def main():
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    test_path = next(INPUT_DIR.glob("*.csv"), None)
    if test_path is None:
        raise FileNotFoundError("Файл test.csv не найден в ./input")

    print("Загружаем тестовые данные...")
    test = pd.read_csv(test_path)

    print("Препроцессинг...")
    test_proc = preprocess(test, freq_maps_path=ARTIFACTS_DIR / "freq_maps.pkl")

    print("Загрузка модели...")
    model, features = load_model_and_features()

    print("Предсказания...")
    preds_proba, preds_class = predict(model, test_proc[features])

    print("Сохранение результатов...")
    save_submission(test, preds_class, OUTPUT_DIR)
    save_importances(model, features, OUTPUT_DIR)
    save_density_plot(preds_proba, OUTPUT_DIR)
    print("Готово! Файлы в /output")

if __name__ == "__main__":
    main()
