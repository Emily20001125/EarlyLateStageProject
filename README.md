# An explainable AI-based prognostic biomarker discovery framework for stage classification

Tumor staging significantly correlates with patient survival and the modern machine learning (ML)-based identification of prognostic genes excels the understanding of the mechanism and clinical implementations. 

Though these ML approaches speed up the discovery, they are also notorius for its well-known "black-box effect", i.e. the lacking of interpretability. Here we proposed a novel explainable pipeline for discovering prognostic biomarkers under cutting-edge ML models in order to identify novel biomarkers, to unravel new understandings of gene functions crossing different stagings and to highlight valuable drug-staging interactions for future development in precision medicine applications. 

Two prevalent renal cancer types, i.e. papillary cell carcinoma (KIRP) and clear cell carcinoma (KIRC) are demonstrated for pipeline construction using transcriptome data from The Cancer Genome Atlas and the pipeline is composed of three major steps:

#Prognostic Feature Selection:
Non-differential gene expression method using optimal patient survival stratification is applied to identify prognostic candidates with subtle expression changes but statistically significant impacts on survival outcomes.
