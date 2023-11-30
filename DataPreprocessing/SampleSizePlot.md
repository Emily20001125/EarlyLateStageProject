Author : Huang,Shu-jing
Date : 2023/08/24.

- [1.1. Environment](#11-environment)
- [1.2. Import package](#12-import-package)
- [1.3. Stage I and Stage II sample size bar plot](#13-stage-i-and-stage-ii-sample-size-bar-plot)
- [1.4. Stage I and Stage II sample size box plot](#14-stage-i-and-stage-ii-sample-size-box-plot)


## 1.1. Environment 
```shell
conda activate python3_10
cd /home/emily2835/stage_project_git2/Sample_size_summary_plot
```
## 1.2. Import package 
```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
```

## 1.3. Stage I and Stage II sample size bar plot
```python
# 1. Read the CSV file with pandas
expr_clin_data = pd.read_csv("Data/Xena_TCGA_Pan_Cancer_com_expr_clin_data.csv",low_memory=False)

# 2. Process the data to count samples for each cancer type and stage
plottable = expr_clin_data.groupby(['cancer type abbreviation', 'stage_group']).size().reset_index(name='n').fillna(0)
# select stage I and stage II
plottable = plottable[plottable['stage_group'].isin(['Stage_I', 'Stage_II'])]
# Pivot the data to get stages as columns
plottable = plottable.pivot(index='cancer type abbreviation', columns='stage_group', values='n').fillna(0)

# 3. Plotting
colors = ['#D66965', '#9BCBEB']  # Add more colors if there are more stages
ax = plottable.plot(kind='bar', stacked=True, color=colors, figsize=(20, 10))

# 5. Annotate bars with the sum of Stage I and Stage II samples
for index, row in plottable.iterrows():
    total_samples = row.sum()
    ax.annotate(f'{int(total_samples)}', 
                (list(plottable.index).index(index), total_samples),
                ha='center', va='bottom',
                color='black', size=10)



# Adjusting the visuals using matplotlib
plt.title("Stage I and Stage II sample size", loc='center')
plt.xlabel("cancer type")
plt.ylabel("sample size")
plt.tight_layout()

# Save the figure
plt.savefig("Figure/Stage_I_II_sample_bar_plot.png", dpi=600)

```

## 1.4. Stage I and Stage II sample size box plot
```python
# read CSV 
expr_clin_data = pd.read_csv("Data/Xena_TCGA_Pan_Cancer_com_expr_clin_data.csv", low_memory=False)

# process the data to count samples for each cancer type and stage
plottable = expr_clin_data.groupby(['cancer type abbreviation', 'stage_group']).size().reset_index(name='n').fillna(0)
# remove stage_unknown
plottable = plottable[plottable['stage_group'] != 'Stage_Unknown']

# plot the bar plot
plt.figure(figsize=(12, 7))
colors = ['#D66965', '#9BCBEB', '#F1E6B2', '#87B18F']
ax =sns.barplot(x='cancer type abbreviation', y='n', hue='stage_group', data=plottable, palette=colors)
# label the number of samples
for p in ax.patches:
    height = p.get_height()
    # Only annotate if height is not NaN
    if not pd.isna(height):
        ax.annotate(f'{int(height)}', 
                    (p.get_x() + p.get_width() / 2., height), 
                    ha='center', va='center', 
                    fontsize=5, color='black', 
                    xytext=(0, 5), 
                    textcoords='offset points')

# set the title and labels
plt.ylabel('Number of Samples')
plt.legend(title='Stage Group')
plt.tight_layout()
plt.savefig("Figure/Total_sample_bar_plot.png",dpi=600)
```

