echo
echo "**************************************** Read dataset ****************************************"
python src/data/make_dataset.py \
	--input_path data/raw/CNNDM_data/train_mini \
	--output_path data/processed/train_mini \
	--maxlen_sentence 25 \
	--maxlen_word 80

python src/data/make_dataset.py \
	--input_path data/raw/CNNDM_data/valid \
	--output_path data/processed/valid \
	--maxlen_sentence 25 \
	--maxlen_word 80
echo

echo "**************************************** Tokenizer dataset ****************************************"
python src/features/convert_data.py \
	--input_path data/processed/train_mini \
	--output_path data/interim/train_mini \
	--save_tokenizer models/tokenizer.json

python src/features/convert_data.py \
	--input_path data/processed/valid \
	--output_path data/interim/valid \
	--save_tokenizer models/tokenizer.json
echo

echo "**************************************** Build and Run model ****************************************"
python src/models/train_model.py \
	--train_folder data/interim/train_mini \
	--valid_folder data/interim/valid \
	--epochs 1 \
	--path_tokenizer models/tokenizer.json \
	--filepath_embedding models/glove.6B.100d.txt \
	--output_model models/model_save.h5 
echo

echo "**************************************** Demo Predict ****************************************"
python src/models/predict_model.py \
	--input_path data/raw/demo_predict.txt \
	--tokenizer_path models/tokenizer.json \
	--model_path models/model_trained.h5
echo