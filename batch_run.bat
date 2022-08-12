


python -m scripts.preperation.generate_data --split=test --n-samples=100 --seed=0
python -m scripts.preperation.generate_data --split=test --n-samples=1000 --seed=0
python -m scripts.preperation.create_bayes_risk_dataset --n-hypotheses=100 --n-references=1000 --split=test


python -m scripts.preperation.create_bayes_risk_dataset --n-hypotheses=100 --n-references=1000 --split=validation_predictive --utility=unigram-f1
python -m scripts.preperation.create_bayes_risk_dataset --n-hypotheses=100 --n-references=1000 --utility=unigram-f1
python -m scripts.preperation.create_bayes_risk_dataset --n-hypotheses=100 --n-references=1000 --split=test --utility=unigram-f1

