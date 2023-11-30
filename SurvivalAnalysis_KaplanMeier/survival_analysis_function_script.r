#import package
#library("tidyverse")
#library("survminer")
#library("survival")
#library("tictoc")


##### survival analysis function script #####
#import data
#file="KIRC_data.csv"
#datafile<-read.csv("KIRC_data.csv",sep=",",row.names=2)
#savefile<-"KIRC_data.csv"



##### suvival rate analysis for all gene #####
#survminer包裡的surv_cutpoint函数，根據biggest/lowest value of the logrank statistics，也就是最小的P值位置，尋找cutoff
#tic()

survival_FUN3 = function(gene){

    res.cut = try(
        surv_cutpoint(
            datafile,
            time = "OS.time",
            event = "OS",
            variables = gene,
            minprop = 0.3,
            progressbar=TRUE
        ),
        silent = TRUE
    )
    if(all(class(res.cut) == "try-error")){
    return(tibble(gene_names=gene, p = NA))}
    

    else{
    res.cat = surv_categorize(res.cut)
    tmp_formula = as.formula(paste0('Surv(OS.time, OS) ~ ', gene))
    fit <- try(surv_fit(tmp_formula, data = res.cat), silent =TRUE)#須將字串轉為formula

        if(all(class(fit) == "try-error")){
        return(tibble(gene_names=gene, p = NA))}
    

        else{
        return(tibble(gene_names=gene, p =surv_pvalue(fit)$pval))}
      } 
 
}

survival_FUN1 = function(gene){

    res.cut = try(
        surv_cutpoint(
            datafile,
            time = "OS.time",
            event = "OS",
            variables = gene,
            minprop = 0.1,
            progressbar=TRUE
        ),
        silent = TRUE
    )
    if(all(class(res.cut) == "try-error")){
    return(tibble(gene_names=gene, p = NA))}
    

    else{
    res.cat = surv_categorize(res.cut)
    tmp_formula = as.formula(paste0('Surv(OS.time, OS) ~ ', gene))
    fit <- try(surv_fit(tmp_formula, data = res.cat), silent =TRUE)#須將字串轉為formula

        if(all(class(fit) == "try-error")){
        return(tibble(gene_names=gene, p = NA))}
    

        else{
        return(tibble(gene_names=gene, p =surv_pvalue(fit)$pval))}
      } 
 
}



#toc()


#data save
surv_pval_save<-function(test_res,savefile){

    #original analysis data save

    survival_pval_table<-do.call(rbind, test_res)
    filename<-str_replace(savefile,pattern = "_data.csv",replacement = "_survival_pval.csv")
    write.table(survival_pval_table,filename,sep=",",row.names = FALSE)

    #survival pvalue<=0.05 selection 
    surv_diffgene <-survival_pval_table %>% filter(survival_pval_table$p <= 0.05)
    diffname<-str_replace(savefile,pattern = "_data.csv", replacement = "_surv_diffgene.csv")
    write.table(surv_diffgene,diffname,sep=",",row.names = FALSE)

    }




surv_pval_save2<-function(test_res,savefile){

    #original analysis data save

    survival_pval_table<-do.call(rbind, test_res)
    filename<-str_replace(savefile,pattern = "_data",replacement = "_RSEM_survival_pval.csv")
    write.table(survival_pval_table,filename,sep=",",row.names = FALSE)

    #survival pvalue<=0.05 selection 
    surv_diffgene <-survival_pval_table %>% filter(survival_pval_table$p <= 0.05)
    diffname<-str_replace(savefile,pattern = "_data", replacement = "_RSEM_surv_diffgene.csv")
    write.table(surv_diffgene,diffname,sep=",",row.names = FALSE)

    }





##### execute example #####


#datafile<-read.csv("KIRC_data.csv",sep=",",row.names=2)
#savefile<-"KIRC_data.csv"

#(input_gene = names(datafile)[35:ncol(datafile)])
#test_res = lapply(1:length(input_gene), function(idx){
#return(survival_FUN(input_gene[idx]))
#})

