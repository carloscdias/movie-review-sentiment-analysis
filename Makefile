# Rules for this project
DATA=data/movie_review_dataset.zip
RAW_DATA_DIRECTORY=build/data/raw
EXPLORE_DATA_DIRECTORY=build/data/explore
ANALYSIS_DATA_DIRECTORY=build/data/analysis

build_raw_data:
ifeq ($(wildcard $(RAW_DATA_DIRECTORY)),)
	echo "Extracting..."
	mkdir -p $(RAW_DATA_DIRECTORY)
	unzip $(DATA) -d $(RAW_DATA_DIRECTORY)
	# create directories
	mkdir $(RAW_DATA_DIRECTORY)/neg
	mkdir $(RAW_DATA_DIRECTORY)/pos
	#rename files
	rename .txt _1.txt $(RAW_DATA_DIRECTORY)/movie_review_dataset/part1/pos/*.txt
	mv $(RAW_DATA_DIRECTORY)/movie_review_dataset/part1/pos/*.txt $(RAW_DATA_DIRECTORY)/pos/
	rename .txt _2.txt $(RAW_DATA_DIRECTORY)/movie_review_dataset/part2/pos/*.txt
	mv $(RAW_DATA_DIRECTORY)/movie_review_dataset/part2/pos/*.txt $(RAW_DATA_DIRECTORY)/pos/
	rename .txt _1.txt $(RAW_DATA_DIRECTORY)/movie_review_dataset/part1/neg/*.txt
	mv $(RAW_DATA_DIRECTORY)/movie_review_dataset/part1/neg/*.txt $(RAW_DATA_DIRECTORY)/neg/
	rename .txt _2.txt $(RAW_DATA_DIRECTORY)/movie_review_dataset/part2/neg/*.txt
	mv $(RAW_DATA_DIRECTORY)/movie_review_dataset/part2/neg/*.txt $(RAW_DATA_DIRECTORY)/neg/
	# Url files
	cat $(RAW_DATA_DIRECTORY)/movie_review_dataset/part1/urls_neg.txt $(RAW_DATA_DIRECTORY)/movie_review_dataset/part2/urls_neg.txt > $(RAW_DATA_DIRECTORY)/urls_neg.txt
	cat $(RAW_DATA_DIRECTORY)/movie_review_dataset/part1/urls_pos.txt $(RAW_DATA_DIRECTORY)/movie_review_dataset/part2/urls_pos.txt > $(RAW_DATA_DIRECTORY)/urls_pos.txt
	rm -rf $(RAW_DATA_DIRECTORY)/movie_review_dataset/
else
	echo "The files were extracted already..."
endif

build_explore_data: build_raw_data
	
clean:
	rm -rf build/data/
