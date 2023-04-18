if [ -n "$1" ]; then
	version="_$1"
	echo "Fetching Test Data Version $1"
else
	version=""
	echo "Fetching Test Data Version 1"
fi

if [ ! -d "/scratch/vis_data$version/test" ]; then
	mkdir -p /scratch/vis_data$version
	rsync -P ada:/share1/$USER/vis_data$version/test.zip /scratch/vis_data$version/
	unzip /scratch/vis_data$version/test.zip -d /scratch/vis_data$version
	rm /scratch/vis_data$version/test.zip
else
	echo "Test Data Already Present"
fi

