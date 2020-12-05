
# Extractive Text Summarization using CNNs

## DATA
CNN/DaliyMail: 
download [here](https://drive.google.com/drive/folders/1s_TuNnStWxEp-1_F-wvLHyZE-CRVFOT2?usp=sharing)

## Dependencies
- python==3.7.9
- jupyter==1.0.0
- numpy==1.19.2
- pandas==1.1.3
- tqdm==4.54.0
- tensorflow==2.1.0
**Install:** 
	Use conda:
	```
		git clone https://github.com/huanhkv/Extractive-Paragraph-Summarization.git
		cd Extractive-Paragraph-Summarization
		conda env create -f env.yml
		conda activate nlp_ats
	```

## Model
This model with input is a paragraph that multi sentences. Maximum sentences in input is 25 sentences and the limit word is 80 each sentence

input shape: (None, 25, 80)

## Commands:
1. **Clean data**
	- input_path: path folder contain input file (`full.txt` and `summ.txt`)
	- output_path: path folder contain output file (`processed_full.txt` and `processed_summ.txt`)
	- maxlen_sentence: this is a number integer
	- maxlen_word: this is a number integer
	```
	python src/data/make_dataset.py 
		--input_path data/raw 
		--output_path data/processed 
		--maxlen_sentence 500 
		--maxlen_word 3000
	```
2. **Tokenizer data**
	- input_path
	- output_path
	- save_tokenizer
	```
	python src/features/convert_data.py 
		--input_path data/processed 
		--output_path data/interim 
		--save_tokenizer models/tokenizer.json
	```
3. **Create and train model**
	- train_folder: 
	- valid_folder: 
	- epochs: 
	- path_tokenizer: 
	- output_model: 
	```	
	python src/models/train_model.py 
		--train_folder data/interim 
		--valid_folder data/interim 
		--epochs 1 
		--path_tokenizer models/tokenizer.json 
		--filepath_embedding models/glove.6B.100d.txt 
		--output_model models/model_save.h5 
	```
4. **Predict**
	- input_path: 
	- output_path:
	- tokenizer_path:
	- model_path:
	```
	python src/models/predict_model.py 
		--input_path data/raw/demo_predict.txt 
		--output_path abc 
		--tokenizer_path models/tokenizer.json 
		--model_path models/model_trained.h5
	```