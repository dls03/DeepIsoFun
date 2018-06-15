########### Get Expression data from read files #####################################################################
########## for both single end and pair end read files ###############################################################

start=1
end=4643

SRAIDs=read.table("sra_filtered_50_60Mreads_5KMB.tsv", header=FALSE, sep="\t",stringsAsFactors=FALSE, fill=TRUE) ## name of all SRA that is downloaded and now processed to get expression data
SRAIDs=SRAIDs[start:end,c(1,7)]
range=start-end+1

it=1;
while(it<=range){

i=SRAIDs[it,1]
len=as.numeric(SRAIDs[it,2])

ids=paste(i,"_1.fastq.gz",sep="");
print(ids);
ids2=paste(i,"_2.fastq.gz",sep="");
print(file.size(ids))
print(file.size(ids2))

############################################### Expression data for pair end reads #################################################
	if(file.exists(ids2)){
			print(ids2);

		fid=paste(i,"_pair",sep="");

		shfilename=paste('kallistoRun',i,'.sh',sep="");

			fileConn<-file(shfilename)
			writeLines(paste("#!/bin/bash -l\n
		#SBATCH --nodes=1\n
		#SBATCH --ntasks=1\n
		#SBATCH --cpus-per-task=1\n
		#SBATCH --mem-per-cpu=2G\n
		#SBATCH --time=0-05:15:00     # 1 day and 15 minutes\n
		#SBATCH --output=my.stdout\n
		#SBATCH --mail-type=ALL\n
		#SBATCH --job-name='kalrun1'\n
		#SBATCH -p intel # This is the default partition, you can use any of the following; intel, batch, highmem, gpu\n
		\n
		\n
		# run kallisto using pair end read\n
		module load kallisto\n
		#kallisto index -i transcripts.idx cds.fasta\n
		kallisto quant -i transcripts.idx --threads=20 -o output_",fid, " -b 100 ", ids, " ", ids2 , sep=""), fileConn)
			close(fileConn);

			system(paste('sbatch',shfilename));
	}
#####################################################################################################################################

############################################# Expression data for single end reads #####################################################
	else{
		fid=paste(i,"_single",sep="");

		shfilename=paste('kallistoRun',i,'.sh',sep="");
			fileConn<-file(shfilename)
			writeLines(paste("#!/bin/bash -l\n
		#SBATCH --nodes=1\n
		#SBATCH --ntasks=1\n
		#SBATCH --cpus-per-task=1\n
		#SBATCH --mem-per-cpu=2G\n
		#SBATCH --time=0-05:15:00     # 1 day and 15 minutes\n
		#SBATCH --output=my.stdout\n
		#SBATCH --mail-type=ALL\n
		#SBATCH --job-name='kalrun1'\n
		#SBATCH -p intel # This is the default partition, you can use any of the following; intel, batch, highmem, gpu\n
		\n
		\n
		# run kallisto using pair end read\n
		module load kallisto\n
		#kallisto index -i transcripts.idx cds.fasta\n
		kallisto quant -i transcripts.idx --threads=20 -o output_",fid, " -b 100 --single -l ", len, " -s 20 ", ids, sep=""), fileConn)
			close(fileConn);
			system(paste('sbatch',shfilename));

	}

it=it+1;
}
####################################################################################################################################
setwd(mydir)


