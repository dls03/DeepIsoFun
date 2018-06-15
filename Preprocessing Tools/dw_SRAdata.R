start=1
end=4643
a=readLines('sra_acc_60_100Mreads_10KMB.tsv') ### Only sra_acc name
b=read.table("sra_filtered_60_100Mreads_10KMB.tsv", header=FALSE, sep="\t",stringsAsFactors=FALSE, fill=TRUE) ### same file as above with more information
sraidv=a[start:end]
b=b[start:end,c(1,7)]
#print(sraidv)
#print(b)

library(systemPipeR)
library(BiocParallel)
moduleload("sratoolkit/2.8.1")
system('fastq-dump --help') # prints help to screen

###################################### Downlaod SRA data ##################################
getSRAfastq <- function(sraid, targetdir, maxreads="1000000000") {
    system(paste("fastq-dump --split-files --gzip --maxSpotId", 
                  maxreads, sraid, "--outdir", targetdir))
}
sessionInfo()
mydir <- getwd(); setwd("./Data/fastq_data/temp_fastq_data/")
bplapply(sraidv, getSRAfastq, targetdir=".", BPPARAM = MulticoreParam(workers=40))
setwd(mydir)
############################################################################################










