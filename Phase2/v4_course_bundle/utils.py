
import pandas as pd
import os

def load_titanic():
    """Load Titanic data from local CSV.
    Tries 'titanic.csv' then 'train.csv' in current dir. Raises if not found.
    Returns a pandas DataFrame.
    """
    for fname in ['titanic_full_dataset.csv', 'train.csv']:
        if os.path.exists(fname):
            df = pd.read_csv(fname)
            return df
    raise FileNotFoundError("Place a Titanic CSV named 'titanic.csv' or 'train.csv' in the same folder as the notebook.")

def print_section(title):
    print("\n" + "="*len(title))
    print(title)
    print("="*len(title))

def basic_eda(df):
    print_section("INFO")
    print(df.info())
    print_section("MISSING VALUES")
    print(df.isna().sum().sort_values(ascending=False))
    print_section("DESCRIBE")
    print(df.describe(include='all').T)

def plot_hist(series, title, xlabel):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6,4))
    series.plot(kind='hist', bins=30)
    plt.xlabel(xlabel); plt.title(title); plt.show()

def bar_from_group(series, title, ylabel="Value", ylim01=False):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6,4))
    series.plot(kind='bar')
    plt.ylabel(ylabel); plt.title(title)
    if ylim01: plt.ylim(0,1)
    plt.show()

def confusion_df(y_true, y_pred):
    from sklearn.metrics import confusion_matrix
    import pandas as pd
    cm = confusion_matrix(y_true, y_pred)
    return pd.DataFrame(cm, index=['Actual 0','Actual 1'], columns=['Pred 0','Pred 1'])
