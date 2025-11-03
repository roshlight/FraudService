# Fraud Detection Inference Service (for Kaggle Competition)

## Что делает
Сервис берёт файл test.csv из ./input и создаёт в ./output:
- sample_submission.csv
- feature_importances_top5.json
- score_density.png

## Как собрать и запустить

docker build -t fraud_service .

поместить в input нужные данные(test.csv)

docker run --rm -v "$(pwd)/input:/app/input" -v "$(pwd)/output:/app/output" fraud_service
