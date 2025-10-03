
# GenAI Forge: Data • Science • Engineering — 12-Week Curriculum

Phase-wise notebooks for Data Engineering, Data Science, and Generative AI. Python is a prerequisite.

## Structure
## Structure
GenAI_Curriculum/
├─ Phase1/
│  ├─ Week1_SQL_NoSQL.ipynb
│  ├─ Week2_ETL.ipynb
│  ├─ Week3_Orchestration.ipynb
│  └─ Week4_Warehouse.ipynb
├─ Phase2/
│  ├─ Week5_EDA_Visualization.ipynb
│  ├─ Week6_Statistics.ipynb
│  ├─ Week7_SupervisedML.ipynb
│  └─ Week8_Unsupervised_Features.ipynb
├─ Phase3/
│  ├─ Week9_DeepLearning.ipynb
│  ├─ Week10_NLP_GenAI.ipynb
│  ├─ Week11_MLOps_Deployment.ipynb
│  └─ Week12_Capstone.ipynb
└─ requirements.txt

## VS Code
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\Activate.ps1
pip install -r requirements.txt
code .
```
In VS Code select the `./venv` interpreter and run notebooks with the Jupyter extension.

## Colab
Upload any notebook and run; install extras if prompted:
```python
# !pip -q install pandas numpy matplotlib seaborn plotly scipy statsmodels scikit-learn duckdb prefect mlflow
```
