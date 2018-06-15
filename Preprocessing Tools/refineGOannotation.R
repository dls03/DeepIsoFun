########## Process Gene annotation ########################################################
gene2go=read.table('gene2gohuman', header=TRUE, sep="\t",stringsAsFactors=FALSE, fill=TRUE)
a=gene2go[gene2go$Evidence!='IEA',] ### remove annotation from IEA
b=a[,2:3]
write.table(b, "GeneAnnoNew", sep="\t", quote=FALSE, sep="\t", quote=FALSE, row.names=FALSE)
############################################################################################






















