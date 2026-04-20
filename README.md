
Рекомендательная система на основе embedding-векторов и matrix factorization.
Для неизвестных значений используются усредненные embedding-вектора.
Для подготовки данных и создания маппинга используется скрипт preprocess_data.

### Данные

| item               | user                  | rating   | timestamp      | user_id | item_id |
|--------------------|-----------------------|----------|----------------|---------|---------|
| B0016LFN2C         | A301F0EVCXRWHU        | 1.0      | 1437350400     | 0       | 0       |
| B00B20OYUO         | A3ME8P7AK6POCR        | 3.0      | 1500595200     | 1       | 1       |
| B00MMLV7VQ         | A3L3E85DXZCWE5        | 4.0      | 1423612800     | 2       | 2       |
| B0006BB9MG         | AFX45TGA12P8K         | 5.0      | 1302912000     | 3       | 3       |


``` docker compose up --build``` 

``` docker compose --profile gpu up --build``` 