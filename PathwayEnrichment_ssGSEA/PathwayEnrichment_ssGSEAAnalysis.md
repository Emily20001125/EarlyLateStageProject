# PathwayEnrichment_ssGSEAAnalysis
Author: Huang,Shu-Jing
Date: 2023-11-28

## Environment
```shell
conda activate ssGSEA
cd /home/emily2835/EarlyLateStageProject/PathwayEnrichment_ssGSEA
```

## Install ssGSEA
```shell
mamba install bioconda::gseapy
```

## Import packages
```python
import pandas as pd
import gseapy as gp
import matplotlib.pyplot as plt
import seaborn as sns
from gseapy.plot import gseaplot
from gseapy import dotplot
```

## Import data
```python
# KIRC
KIRClist = pd.read_csv("Data/KIRCearlylatelabelCoxProggene005.csv",index_col=0)
EarlyKIRC = pd.read_csv("Data/KIRCearlystageExprandClin.csv",index_col=0)
LateKIRC = pd.read_csv("Data/KIRClatestageExprandClin.csv",index_col=0)
compare = pd.read_csv("Data/probeMap_gencode.v23.annotation.gene.probemap",sep="\t")
compare["id"] = compare["id"].str.split(".").str[0]
# KIRP
KIRPlist = pd.read_csv("Data/KIRPearlylatelabelCoxProggene005.csv",index_col=0)
EarlyKIRP = pd.read_csv("Data/KIRPearlystageExprandClin.csv",index_col=0)
LateKIRP = pd.read_csv("Data/KIRPlatestageExprandClin.csv",index_col=0)
compare = pd.read_csv("Data/probeMap_gencode.v23.annotation.gene.probemap",sep="\t")
compare["id"] = compare["id"].str.split(".").str[0]
```

## Select the earlyandlate gene expression data
```python
# KIRC
# select the earlyandlate gene
KIRClist = KIRClist[KIRClist["ProgStage"] == "earlylate"]
# select the earlyandlate gene expression data
EarlyKIRCSub = EarlyKIRC[KIRClist.index]
LateKIRCSub = LateKIRC[KIRClist.index]
# Add sample name a column name
EarlyKIRCSub.index = EarlyKIRC["sample"]
LateKIRCSub.index= LateKIRC["sample"]
# Change the column name from ensembl to gene symbol
EarlyKIRCSub = EarlyKIRCSub.rename(columns=compare.set_index("id")["gene"])
LateKIRCSub = LateKIRCSub.rename(columns=compare.set_index("id")["gene"])
# Transpose the dataframe
EarlyKIRCSub = EarlyKIRCSub.T
LateKIRCSub = LateKIRCSub.T
# Combine the earlyandlate gene expression data
KIRCSub = pd.concat([EarlyKIRCSub,LateKIRCSub],axis=1)
# Cheack the duplicated gene symbol
KIRCSub[KIRCSub.index.duplicated()]
# Remove the duplicated gene symbol
KIRCSub = KIRCSub[~KIRCSub.index.duplicated()]

# KIRP
# select the earlyandlate gene
KIRPlist = KIRPlist[KIRPlist["ProgStage"] == "earlylate"]
# select the earlyandlate gene expression data
EarlyKIRPSub = EarlyKIRP[KIRPlist.index]
LateKIRPSub = LateKIRP[KIRPlist.index]
# Add sample name a column name
EarlyKIRPSub.index = EarlyKIRP["sample"]
LateKIRPSub.index= LateKIRP["sample"]
# Change the column name from ensembl to gene symbol
EarlyKIRPSub = EarlyKIRPSub.rename(columns=compare.set_index("id")["gene"])
LateKIRPSub = LateKIRPSub.rename(columns=compare.set_index("id")["gene"])
# Transpose the dataframe
EarlyKIRPSub = EarlyKIRPSub.T
LateKIRPSub = LateKIRPSub.T
# Combine the earlyandlate gene expression data
KIRPSub = pd.concat([EarlyKIRPSub,LateKIRPSub],axis=1)
# Cheack the duplicated gene symbol
KIRPSub[KIRPSub.index.duplicated()]
# Remove the duplicated gene symbol
KIRPSub = KIRPSub[~KIRPSub.index.duplicated()]
```

## KIRC ssGSEA analysis
### GSEA
```python
# Create the sample vector with "Early" - 321 and "Late" - 207 
class_vector = ["Early"]*321 + ["Late"]*207
# gsea analysis
gs_res = gp.gsea(data = KIRCSub, 
                 gene_sets='Data/h.all.v2023.2.Hs.symbols.gmt', # choose the gene set
                 cls= class_vector, # cls = class_vector
                 # set permutation_type to phenotype if samples >= 15
                 permutation_type='phenotype',
                 permutation_num=1000, # reduce number to speed up test
                 outdir=None,  # do not write output to disk
                 method='signal_to_noise',
                 threads=4, seed= 7)

# save the result
result = gs_res.res2d
```
### Select the FDR < 0.25 and |NES| > 1
```python
# FDR < 0.25 and |NES| > 1
result = result[(result["FDR q-val"] < 0.25) & (abs(result["NES"]) > 1)]
# Sort the NES
result = result.sort_values(by="NES",ascending=False)
```
### Plot the top 10 pathways
```python
# Plot the top 5 pathways
terms = result.Term
axs = gs_res.plot(terms[:10], show_ranking=False, legend_kws={'loc': (1.05, 0)}, figsize=(10,15))
plt.savefig("Figure/KIRCTop10Pathways.png",dpi=300,bbox_inches='tight')
plt.close()
```
### Function `singlepathways_plot` : plot the single pathway
```python
# Plot the pathways
def singlepathways_plot(gs_res, term_index):
    terms = result.Term
    # reset the index
    terms = terms.reset_index(drop=True)
    terms.index = terms.index + 1
    axs = gs_res.plot(terms=terms[term_index]) 
    plt.savefig("Figure/KIRC"+terms[term_index]+".png",dpi=300,bbox_inches='tight')
    plt.close()

singlepathways_plot(gs_res, 1)
singlepathways_plot(gs_res, 2)
singlepathways_plot(gs_res, 3)
singlepathways_plot(gs_res, 4)
singlepathways_plot(gs_res, 5)
singlepathways_plot(gs_res, 6)
singlepathways_plot(gs_res, 7)
singlepathways_plot(gs_res, 8)
singlepathways_plot(gs_res, 9)
singlepathways_plot(gs_res, 10)
```
### plot the dotplot
```python
ax = dotplot(result,
             column="FDR q-val",
             title='HALLMARK',
             cmap=plt.cm.plasma,
             size=5,
             figsize=(5,10), cutoff=1)
plt.savefig("Figure/KIRCHALLMARKDotplot.png",dpi=300,bbox_inches='tight')
```




## KIRP ssGSEA analysis
### GSEA
```python
# Create the sample vector with "Early" - 189 and "Late" - 96
class_vector = ["Early"]*189 + ["Late"]*96
# gsea analysis
gs_res = gp.gsea(data = KIRPSub, 
                 gene_sets='Data/h.all.v2023.2.Hs.symbols.gmt', # choose the gene set
                 cls= class_vector, # cls = class_vector
                 # set permutation_type to phenotype if samples >= 15
                 permutation_type='phenotype',
                 permutation_num=1000, # reduce number to speed up test
                 outdir=None,  # do not write output to disk
                 method='signal_to_noise',
                 threads=4, seed= 7)

# save the result
result = gs_res.res2d
```
### Select the FDR < 0.25 and |NES| > 1
```python
# FDR < 0.25 and |NES| > 1
result = result[(result["FDR q-val"] < 0.25) & (abs(result["NES"]) > 1)]
# Sort the NES
result = result.sort_values(by="NES",ascending=False)
```
### Plot the top 10 pathways
```python
# Plot the top 5 pathways
terms = result.Term
axs = gs_res.plot(terms[:5], show_ranking=False, legend_kws={'loc': (1.05, 0)}, figsize=(10,15))
plt.savefig("Figure/KIRPTop10Pathways.png",dpi=300,bbox_inches='tight')
plt.close()
```
### Function `singlepathways_plot` : plot the single pathway
```python
# Plot the pathways
def singlepathways_plot(gs_res, term_index):
    terms = result.Term
    # reset the index
    terms = terms.reset_index(drop=True)
    terms.index = terms.index + 1
    axs = gs_res.plot(terms=terms[term_index]) 
    plt.savefig("Figure/KIRP"+terms[term_index]+".png",dpi=300,bbox_inches='tight')
    plt.close()

singlepathways_plot(gs_res, 1)
singlepathways_plot(gs_res, 2)
singlepathways_plot(gs_res, 3)
singlepathways_plot(gs_res, 4)
singlepathways_plot(gs_res, 5)
```
### plot the dotplot
```python
ax = dotplot(result,
             column="FDR q-val",
             title='HALLMARK',
             cmap=plt.cm.plasma,
             size=5,
             figsize=(5,10), cutoff=1)
plt.savefig("Figure/KIRPHALLMARKDotplot.png",dpi=300,bbox_inches='tight')
