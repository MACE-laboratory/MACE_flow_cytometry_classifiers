#python3 train_pairwise_ensemble_vfinal.py -r 1
#python3 apply_pairwise_ensemble_to_cocultures_vfinal.py -r 1

#python3 train_pairwise_ensemble_vfinal.py -r 2
#python3 apply_pairwise_ensemble_to_cocultures_vfinal.py -r 2

#python3 train_pairwise_ensemble_vfinal.py -r 3
#python3 apply_pairwise_ensemble_to_cocultures_vfinal.py -r 3

python validate_models.py --n-train 13000 --n-test 4000 --skip-train --output-csv model_validation_results.csv --data-dir ../data/classifier_test/
