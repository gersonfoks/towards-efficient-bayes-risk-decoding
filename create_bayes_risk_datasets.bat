





python -m scripts.preperation.create_bayes_risk_dataset --n-hypotheses=10 --n-references=100 --smoke-test
python -m scripts.preperation.create_bayes_risk_dataset --n-hypotheses=100 --n-references=1000 --smoke-test
python -m scripts.preperation.create_bayes_risk_dataset --n-hypotheses=10 --n-references=100 --smoke-test --split=validation_predictive
python -m scripts.preperation.create_bayes_risk_dataset --n-hypotheses=100 --n-references=1000 --smoke-test --split=validation_predictive





python -m scripts.preperation.create_bayes_risk_dataset --n-hypotheses=10 --n-references=100 --split=validation_predictive
python -m scripts.preperation.create_bayes_risk_dataset --n-hypotheses=100 --n-references=1000 --split=validation_predictive
python -m scripts.preperation.create_bayes_risk_dataset --n-hypotheses=10 --n-references=100
python -m scripts.preperation.create_bayes_risk_dataset --n-hypotheses=100 --n-references=1000




