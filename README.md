# Codebase for "Generative Adversarial Imputation Networks (GAIN)"

To run the pipeline for training and evaluation on GAIN framwork, simply run 
python3 -m main_letter_spam.py.

### Command inputs:

-   data_name: letter or spam
-   miss_rate: probability of missing components
-   batch_size: batch size
-   hint_rate: hint rate
-   alpha: hyperparameter
-   iterations: iterations

### Example command

python main_letter_spam.py --data_name pmu --miss_rate 0.2 --batch_size 128 --hint_rate 0.9 --alpha 100 --iterations 10000

### Outputs

-   imputed_data_x: imputed data
-   rmse: Root Mean Squared Error
