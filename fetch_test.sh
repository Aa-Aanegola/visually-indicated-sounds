# if [ ! -d "/scratch/vis_data/test" ]; then
mkdir -p /scratch/vis_data
rsync -P ada:/share1/$USER/vis_data/test.zip /scratch/vis_data/
unzip /scratch/vis_data/test.zip -d /scratch/vis_data
rm /scratch/vis_data/test.zip
# else
	# echo "Test Data Already Present"
# fi

