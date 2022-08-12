
python -m scripts.preperation.generate_data --split=validation_predictive --n-samples=10 --seed=1
python -m scripts.preperation.generate_data --split=validation_predictive --n-samples=5 --seed=1


python -m scripts.preperation.create_bayes_risk_dataset --n-hypotheses=100 --n-references=5 --split=validation_predictive --seed=1
python -m scripts.preperation.create_bayes_risk_dataset --n-hypotheses=100 --n-references=10 --split=validation_predictive --seed=1

