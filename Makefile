# Rules for this project
DATA=data/movie_review_dataset.zip
RAW_DATA_DIRECTORY=build/data/raw
EXPLORE_DATA_DIRECTORY=build/data/explore
ANALYSIS_DATA_DIRECTORY=build/data/analysis
EXPLORE_DATA_LEN=5000

all: generate_25_unigram_features_dataset generate_50_unigram_features_dataset generate_25_bigram_features_dataset generate_50_bigram_features_dataset

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
ifeq ($(wildcard $(EXPLORE_DATA_DIRECTORY)),)
	echo "Copying random $(EXPLORE_DATA_LEN) raw files to build explore data..."
	mkdir -p $(EXPLORE_DATA_DIRECTORY)/neg
	mkdir -p $(EXPLORE_DATA_DIRECTORY)/pos
	ls $(RAW_DATA_DIRECTORY)/pos/* | shuf | head -n $(EXPLORE_DATA_LEN) | xargs cp -t $(EXPLORE_DATA_DIRECTORY)/pos/
	ls $(RAW_DATA_DIRECTORY)/neg/* | shuf | head -n $(EXPLORE_DATA_LEN) | xargs cp -t $(EXPLORE_DATA_DIRECTORY)/neg/
	echo "Explore data generated"
else
	echo "Files already in the explore directory..."
endif

generate_25_unigram_features_dataset: build_raw_data
	src/extractor.py -g 1 -f 25 $(RAW_DATA_DIRECTORY)

generate_50_unigram_features_dataset: build_raw_data
	src/extractor.py -g 1 -f 50 $(RAW_DATA_DIRECTORY)

generate_25_bigram_features_dataset: build_raw_data
	src/extractor.py -g 2 -f 25 $(RAW_DATA_DIRECTORY)

generate_50_bigram_features_dataset: build_raw_data
	src/extractor.py -g 2 -f 50 $(RAW_DATA_DIRECTORY)

clean:
	rm -rf build/data/

