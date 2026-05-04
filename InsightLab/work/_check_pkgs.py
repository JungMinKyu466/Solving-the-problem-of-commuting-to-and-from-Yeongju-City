import importlib
for pkg in ['numpy','pandas','sklearn','lightgbm','xgboost','catboost','shap','matplotlib']:
    try:
        m = importlib.import_module(pkg)
        v = getattr(m, '__version__', '?')
        print(f'  {pkg}: {v}')
    except ImportError:
        print(f'  {pkg}: NOT INSTALLED')
