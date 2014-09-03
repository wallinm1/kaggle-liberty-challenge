kaggle-liberty-challenge
========================

Script for Kaggle's Liberty Challenge contest.

Instructions for generating submission files:

1. Download the .csv-files `sampleSubmission.csv`, `test.csv` and `train.csv` into the `data`-folder of this repository.

2. Run the script `train.py`
	- If the `build_features`-flag is manually set to `False`, the script reads the `feats`-dictionary from the `features`-folder and will finish faster.
	- Otherwise, the features are built up from scratch.

3. The `train.py`-script will generate the submission files `gbm_sub.csv` and `mean_sub.csv` in the `submissions` folder.
	- `gbm_sub.csv`scores 0.32160 on the private leaderboard and `mean_sub.csv` scores 0.32464 (i.e. 4th place for both)

4. Most of the computation time is spent training the gb-regressor.
	- On an i7-4790, training the randomized lasso feature selector takes around 10 minutes, training the L1-feature selectors take around 20 minutes, and training the gb-regressor take about 70 minutes.
	- The script uses a lot of RAM in in preprocessing. 
	- Also during the training of the randomized lasso feature selector, RAM usage occasionally hits 100% on a 16 GB machine, along with periodical sharp rises in CPU temps. This RAM usage can be lowered by reducing the n_jobs-parameter of the randomized lasso routine.

5. It should be noted that these submissions are not the ones I submitted to the contest. I moved towards more complex models which performed better on the public leaderboard but much worse on the private leaderboard.
	- I figured that this model was interesting as it did quite well on the private leaderboard while being very simple.
	- The 4th place finish can be achieved with one single gradient boosting regressor fitted on a selection of roughly 20 features.
		