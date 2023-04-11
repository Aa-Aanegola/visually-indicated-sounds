if [ ! -d "/scratch/vis_data/train" ]; then
	mkdir -p /scratch/vis_data
	rsync -P ada:/share1/$USER/vis_data/train.zip /scratch/vis_data/
	unzip /scratch/vis_data/train.zip -d /scratch/vis_data/train
	rm /scratch/vis_data/train.zip
else
	echo "Train Data Already Present"
fi

