| EXP 1   | TP    | TN     | FP    | FN    | Support | Sensitivity | Specificity | Balanced Accuracy | F1 | Precision | Macro F1 |
|---------|-------|--------|-------|-------|---------|-------------|-------------|-------------------|----|-----------|----------|
| Normal  | 25788 | 206445 | 51    | 24    | 25812   | 1.00        | 1.00        | 1.00              |1.00|   1.00    |   1.00   |
| RBBB    | 25812 | 206496 | 0     | 0     | 25812   | 1.00        | 1.00        | 1.00              |1.00|   1.00    |   1.00   |
| PVC     | 25808 | 206496 | 0     | 4     | 25812   | 1.00        | 1.00        | 1.00              |1.00|   1.00    |   1.00   |
| FUSION  | 25812 | 206493 | 3     | 0     | 25812   | 1.00        | 1.00        | 1.00              |1.00|   1.00    |   1.00   |
| APC     | 25762 | 206488 | 8     | 50    | 25812   | 1.00        | 1.00        | 1.00              |1.00|   1.00    |   1.00   |
| SVPB    | 25812 | 206490 | 6     | 0     | 25812   | 1.00        | 1.00        | 1.00              |1.00|   1.00    |   1.00   |
| NESC    | 25812 | 206496 | 0     | 0     | 25812   | 1.00        | 1.00        | 1.00              |1.00|   1.00    |   1.00   |
| UNKNOWN | 25812 | 206494 | 2     | 0     | 25812   | 1.00        | 1.00        | 1.00              |1.00|   1.00    |   1.00   |
| SVESC   | 25812 | 206488 | 8     | 0     | 25812   | 1.00        | 1.00        | 1.00              |1.00|   1.00    |   1.00   |
| Total   |       |        |       |       | 232308  |             |             |                   |    |           |          |


| EXP 2   | TP    | TN     | FP    | FN    | Support | Sensitivity | Specificity | Balanced Accuracy | F1 | Precision | Macro F1 |
|---------|-------|--------|-------|-------|---------|-------------|-------------|-------------------|----|-----------|----------|
| Normal  | 25768 | 5115   | 16    | 44    | 25812   | 1.00        | 1.00        | 1.00              |1.00|  1.00     | 0.85     |
| RBBB    | 642   | 30301  | 0     | 0     | 642     | 1.00        | 1.00        | 1.00              |1.00|  1.00     |          |
| PVC     | 3976  | 26951  | 3     | 13    | 3989    | 1.00        | 1.00        | 1.00              |1.00|  1.00     |          |
| FUSION  | 81    | 30857  | 5     | 0     | 81      | 1.00        | 1.00        | 1.00              |0.97|  0.94     |          |
| APC     | 390   | 30536  | 12    | 5     | 395     | 0.99        | 1.00        | 0.99              |0.98|  0.97     |          |
| SVPB    | 3     | 30930  | 10    | 0     | 3       | 1.00        | 1.00        | 1.00              |0.38|  0.23     |          |
| NESC    | 13    | 30930  | 0     | 0     | 13      | 1.00        | 1.00        | 1.00              |1.00|  1.00     |          |
| UNKNOWN | 4     | 30939  | 0     | 0     | 4       | 1.00        | 1.00        | 1.00              |1.00|  1.00     |          |
| SVESC   | 4     | 30923  | 16    | 0     | 4       | 1.00        | 1.00        | 1.00              |0.33|  0.20     |          |
|  Total  |       |        |       |       | 30943   |             |             |                   |    |           |          |


| EXP 3   | TP    | TN     | FP    | FN    | Support | Sensitivity | Specificity | Balanced Accuracy | F1   | Precision |  | Macro F1 |
|---------|-------|--------|-------|-------|---------|------|------|--------------|------|-----------|--|----------|
| Normal  | 25806 | 149914 | 56582 | 6     | 25812   | 1.00 | 0.73 | 0.86         | 0.48 | 0.31      |  | 0.66     |
| RBBB    | 25810 | 206315 | 181   | 2     | 25812   | 1.00 | 1.00 | 1.00         | 1.00 | 0.99      |  |          |
| PVC     | 25748 | 201097 | 5399  | 64    | 25812   | 1.00 | 0.97 | 0.99         | 0.90 | 0.83      |  |          |
| FUSION  | 23152 | 202172 | 4324  | 2660  | 25812   | 0.90 | 0.98 | 0.94         | 0.87 | 0.84      |  |          |
| APC     | 24234 | 203043 | 3453  | 1578  | 25812   | 0.94 | 0.98 | 0.96         | 0.91 | 0.88      |  |          |
| SVPB    | 0     | 206496 | 0     | 25812 | 25812   | 0.00 | 1.00 | 0.50         | 0.00 | N/A       |  |          |
| NESC    | 22453 | 206496 | 0     | 3359  | 25812   | 0.87 | 1.00 | 0.93         | 0.93 | 1.00      |  |          |
| UNKNOWN | 10617 | 206496 | 0     | 15195 | 25812   | 0.41 | 1.00 | 0.71         | 0.58 | 1.00      |  |          |
| SVESC   | 4549  | 206496 | 0     | 21263 | 25812   | 0.18 | 1.00 | 0.59         | 0.30 | 1.00      |  |          |
|   Total |       |        |       |       | 232308  |      |      |              |      |           |  |          |


| EXP 4   | TP    | TN     | FP    | FN    | Support | Sensitivity | Specificity | Balanced Accuracy | F1   | Precision |  | Macro F1 |
|---------|-------|--------|-------|-------|---------|------|------|--------------|------|-----------|--|----------|
| Normal  | 25799 | 5079   | 52    | 13    | 25812   | 1.00 | 0.99 | 0.99         | 1.00 | 1.00      |  | 0.68     |
| RBBB    | 642   | 30301  | 0     | 0     | 642     | 1.00 | 1.00 | 1.00         | 1.00 | 1.00      |  |          |
| PVC     | 3976  | 26944  | 10    | 13    | 3989    | 1.00 | 1.00 | 1.00         | 1.00 | 1.00      |  |          |
| FUSION  | 69    | 30849  | 13    | 12    | 81      | 0.85 | 1.00 | 0.93         | 0.85 | 0.84      |  |          |
| APC     | 366   | 30545  | 3     | 29    | 395     | 0.93 | 1.00 | 0.96         | 0.96 | 0.99      |  |          |
| SVPB    | 0     | 30940  | 0     | 3     | 3       | 0.00 | 1.00 | 0.50         | 0.00 | N/A       |  |          |
| NESC    | 11    | 30929  | 1     | 2     | 13      | 0.85 | 1.00 | 0.92         | 0.88 | 0.92      |  |          |
| UNKNOWN | 1     | 30939  | 0     | 3     | 4       | 0.25 | 1.00 | 0.63         | 0.40 | 1.00      |  |          |
| SVESC   | 0     | 30939  | 0     | 4     | 4       | 0.00 | 1.00 | 0.50         | 0.00 | N/A       |  |          |
|   Total |       |        |       |       | 30943   |      |      |              |      |           |  |          |

(N/A means there is no true positive for the class and, therefore, the Precision is undefined.)