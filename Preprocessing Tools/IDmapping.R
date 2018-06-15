GRA=read.table("HUMAN_9606_idmapping_selected.tab", skip=0, header=FALSE, sep="\t",stringsAsFactors=FALSE, fill=TRUE)
RtRp=read.table("LRG_RefSeqGene", skip=0, header=FALSE, sep="\t",stringsAsFactors=FALSE, fill=TRUE)
Rt=read.table("RefSecID", skip=0, header=FALSE, sep="\t",stringsAsFactors=FALSE, fill=TRUE)

### Select only those column that are necessary for ID mapping#######
GRA3=GRA[,c(3,4)]
RtRp3=RtRp[,c(2,6,8)]
######################################################################

################ merge files ###########################################
colnames(GRA3)[1]<-'geneID'
colnames(RtRp3)[1]<-'geneID'
amerge=merge(GRA3, RtRp3, by="geneID", all.y=TRUE)
colnames(amerge)[3]<-'RefSecID'
colnames(Rt)[2]<-'RefSecID'
bmerge=merge(Rt, amerge, by="RefSecID", all.x=TRUE)
########################################################################

############## remove duplicate rows ################################
GIName=bmerge[c(1,3)]
GIName = GIName[!duplicated(GIName$RefSecID),]
write.table(GIName, file="GeneIsoformName", sep="\t", row.names = FALSE, col.names = TRUE)
#############################################################################

############################## convert to dataframe and save ##########################
GIName=read.table('GeneIsoformName', header=TRUE, sep="\t",stringsAsFactors=FALSE, fill=TRUE)
GINameo=GIName[order(GIName$GeneID),]
head(GINameo)
df <- transform(GINameo, isoformID=match(RefSeqID, unique(RefSeqID)))
df2 <- transform(df, geneID=match(GeneID, unique(GeneID)))
write.table(df2, "GeneIsoformNameNew", sep="\t", quote=FALSE, row.names=FALSE)
######################################################################################


























