





python -m scripts.preperation.generate_data --split=train_predictive --n-samples=10 --smoke-test --seed=0
python -m scripts.preperation.generate_data --split=train_predictive --n-samples=100 --smoke-test --seed=0
python -m scripts.preperation.generate_data --split=train_predictive --n-samples=1000 --smoke-test --seed=0

python -m scripts.preperation.generate_data --split=validation_predictive --n-samples=10 --smoke-test --seed=0
python -m scripts.preperation.generate_data --split=validation_predictive --n-samples=100 --smoke-test --seed=0
python -m scripts.preperation.generate_data --split=validation_predictive --n-samples=1000 --smoke-test --seed=0

python -m scripts.preperation.generate_data --split=train_predictive --n-samples=10 --seed=0
python -m scripts.preperation.generate_data --split=train_predictive --n-samples=100 --seed=0
python -m scripts.preperation.generate_data --split=train_predictive --n-samples=1000 --seed=0

python -m scripts.preperation.generate_data --split=validation_predictive --n-samples=10 --seed=0
python -m scripts.preperation.generate_data --split=validation_predictive --n-samples=100 --seed=0
python -m scripts.preperation.generate_data --split=validation_predictive --n-samples=1000 --seed=0



