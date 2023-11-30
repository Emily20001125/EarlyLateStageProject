# Prognostic Gene Summary
Author: Huang,Shu-Jing
Date: 2023-11-24

## environment
```shell
conda activate python3_10
cd /home/emily2835/stage_project_git3/SurvivalAnalysis_CoxRegression/CoxProggeneSummary
```
## Import packages
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib_venn import venn2, venn3
from matplotlib import pyplot as plt
```

## Function`plot_venn_diagram` : intersect of earlyProg and lateProg
```python
def plot_venn_diagram(cancer_type):
    # read data
    early_stage_data = pd.read_csv(f'Data/{cancer_type}earlystageCoxProggeneP005.csv', index_col=0)
    late_stage_data = pd.read_csv(f'Data/{cancer_type}latestageCoxProggeneP005.csv', index_col=0)
    plt.figure(figsize=(20,16))  
    # Plot Venn diagram for earlyProg and lateProg rownames
    venn2(subsets=[set(early_stage_data.index), set(late_stage_data.index)], 
          set_labels=('Early', 'Late'),
          set_colors=["#236192", "#B33D26"])
    # font size change
    plt.rcParams.update({'font.size': 30})
    # Add title and annotation
    plt.title(f'The intersect of early stage and late stage prognostic genes for {cancer_type}', fontsize=35)
    plt.savefig(f'Figure/{cancer_type}earlylateProgVenn.png', dpi=300)
    plt.savefig(f'Figure/{cancer_type}earlylateProgVenn.svg')

for i in ['COAD', 'KIRC', 'KIRP', 'LIHC', 'LUAD', 'LUSC', 'STAD','ESCA']:
    plot_venn_diagram(i)
```



## Function`plot_venn_diagram` :Separate data into earlyOnly, lateOnly, earlylate
## Import data
```python
def Separatedataearlylate(cancer_type):
    # read data
    early_stage_file = f'Data/{cancer_type}earlystageCoxProggeneP005.csv'
    late_stage_file = f'Data/{cancer_type}latestageCoxProggeneP005.csv'
    early_stage_data = pd.read_csv(early_stage_file, index_col=0)
    late_stage_data = pd.read_csv(late_stage_file, index_col=0)
    # add gene column
    early_stage_data['gene'] = early_stage_data.index
    late_stage_data['gene'] = late_stage_data.index
    # separate data into earlyOnly, lateOnly, earlylate
    early_only_data = early_stage_data[~early_stage_data['gene'].isin(late_stage_data['gene'])]
    early_only_data['ProgStage'] = 'earlyOnly'
    late_only_data = late_stage_data[~late_stage_data['gene'].isin(early_stage_data['gene'])]
    late_only_data['ProgStage'] = 'lateOnly'
    early_late_data = early_stage_data[early_stage_data['gene'].isin(late_stage_data['gene'])]
    early_late_data['ProgStage'] = 'earlylate'
    # check the number of genes in each group
    print(len(early_only_data), len(late_only_data), len(early_late_data))
    # combine data
    combined_data = pd.concat([early_only_data, late_only_data, early_late_data])
    # save data
    output_file = f'Output/{cancer_type}earlylatelabelCoxProggene005.csv'
    combined_data.to_csv(output_file)

for i in ['COAD', 'KIRC', 'KIRP', 'LIHC', 'LUAD', 'LUSC', 'STAD','ESCA']:
    Separatedataearlylate(i)
```

