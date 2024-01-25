# Kaplan Meier Survival Analysis for Stage
## Environment Setup
```python
conda activate python3_10
cd /home/emily2835/stage_project_git3/SurvivalAnalysis_KaplanMeier
```
## Import packages
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from matplotlib.lines import Line2D
```


## Function`survival_analysis`: for Kaplan Meier Survival Analysis
```python
def survival_analysis(cancer_type):
    # Load the data
    early_stage_data = pd.read_csv(f"Data/{cancer_type}earlystageExprandClin.csv")
    late_stage_data = pd.read_csv(f"Data/{cancer_type}latestageExprandClin.csv")
    # Combine the two dataframes
    data = pd.concat([early_stage_data, late_stage_data])
    # Change the OS.time column to months
    data['OS.time'] = data['OS.time']/30
    # Extract the OS and OS.time columns
    data = data[['OS', 'OS.time','E_L_Stage']]
    # Separate the data into early and late stages
    early_stage_data = data[data['E_L_Stage'] == 'early_stage']
    late_stage_data = data[data['E_L_Stage'] == 'late_stage']
    # Perform the log-rank test
    results = logrank_test(early_stage_data['OS.time'], late_stage_data['OS.time'], 
                           event_observed_A=early_stage_data['OS'], event_observed_B=late_stage_data['OS'])
    # Get p-value
    p_value = results.p_value
    p_value = "{:.2e}".format(p_value)
    print(f'p-value for {cancer_type}: {p_value}')
    # Initialize the KaplanMeierFitter model
    kmf = KaplanMeierFitter()
    # Fit the data into the model for early stage
    kmf.fit(early_stage_data['OS.time'], early_stage_data['OS'], label=f'early_stage (n={len(early_stage_data)})')
    ax = kmf.plot_survival_function(ci_show=False, color='#236192')
    # Fit the data into the model for late stage
    kmf.fit(late_stage_data['OS.time'], late_stage_data['OS'], label=f'late_stage (n={len(late_stage_data)})')
    kmf.plot_survival_function(ax=ax, ci_show=False, color='#B33D26')
    # Create a custom legend
    legend_elements = [Line2D([0], [0], color='#236192', lw=2, label=f'early_stage (n={len(early_stage_data)})'),
                       Line2D([0], [0], color='#B33D26', lw=2, label=f'late_stage (n={len(late_stage_data)})'),
                       Line2D([0], [0], color='white', label=f'p-value: {p_value}')]
    plt.legend(handles=legend_elements)
    # Create an estimate
    plt.title(f'Survival Analysis of {cancer_type} Across Different Stages')
    plt.xlabel('Time in months')
    plt.ylabel('Survival probability')
    plt.savefig(f'Figure/{cancer_type}earlyandlateKMplot.png')
    plt.close()

# Call the function
for cancer_type in ['KIRP', 'KIRC','COAD','STAD','LUAD','LUSC','LIHC','ESCA']:
    survival_analysis(cancer_type)
```



## Function`survival_analysis_eachstage`: for Kaplan Meier Survival Analysis
```python
def survival_analysis_eachstage(cancer_type):
    # Load the data
    early_stage_data = pd.read_csv(f"Data/{cancer_type}earlystageExprandClin.csv")
    late_stage_data = pd.read_csv(f"Data/{cancer_type}latestageExprandClin.csv")
    # Combine the two dataframes
    data = pd.concat([early_stage_data, late_stage_data])
    # Change the OS.time column to months
    data['OS.time'] = data['OS.time']/30
    # Extract the OS and OS.time columns
    data = data[['OS', 'OS.time','stage_group']]
    # Separate the data into early and late stages
    Stage_I_data = data[data['stage_group'] == 'Stage_I']
    Stage_II_data = data[data['stage_group'] == 'Stage_II']
    Stage_III_data = data[data['stage_group'] == 'Stage_III']
    Stage_IV_data = data[data['stage_group'] == 'Stage_IV']
    # Initialize the KaplanMeierFitter model
    kmf = KaplanMeierFitter()
    # Fit the data into the model for stage I
    kmf.fit(Stage_I_data['OS.time'], Stage_I_data['OS'], label=f'Stage_I (n={len(Stage_I_data)})')
    ax = kmf.plot_survival_function(ci_show=False, color='#236192')
    # Fit the data into the model for stage II
    kmf.fit(Stage_II_data['OS.time'], Stage_II_data['OS'], label=f'Stage_II (n={len(Stage_II_data)})')
    kmf.plot_survival_function(ax=ax, ci_show=False, color='#789F90')
    # Fit the data into the model for stage III
    kmf.fit(Stage_III_data['OS.time'], Stage_III_data['OS'], label=f'Stage_III (n={len(Stage_III_data)})')
    kmf.plot_survival_function(ax=ax, ci_show=False, color='#DE754C')
    # Fit the data into the model for stage IV
    kmf.fit(Stage_IV_data['OS.time'], Stage_IV_data['OS'], label=f'Stage_IV (n={len(Stage_IV_data)})')
    kmf.plot_survival_function(ax=ax, ci_show=False, color='#B33D26')
    # Create a custom legend
    legend_elements = [Line2D([0], [0], color='#236192', lw=2, label=f'Stage_I (n={len(Stage_I_data)})'),
                       Line2D([0], [0], color='#789F90', lw=2, label=f'Stage_II (n={len(Stage_II_data)})'),
                       Line2D([0], [0], color='#DE754C', lw=2, label=f'Stage_III (n={len(Stage_III_data)})'),
                       Line2D([0], [0], color='#B33D26', lw=2, label=f'Stage_IV (n={len(Stage_IV_data)})')]
    plt.legend(handles=legend_elements)
    # Create an estimate
    plt.title(f'Survival Analysis of {cancer_type} Across Different Stages')
    plt.xlabel('Time in months')
    plt.ylabel('Survival probability')
    plt.savefig(f'Figure/{cancer_type}eachstageKMplot.png')
    plt.close()

# Call the function
for cancer_type in ['KIRP', 'KIRC','COAD','STAD','LUAD','LUSC','LIHC','ESCA']:
    survival_analysis_eachstage(cancer_type)
```

