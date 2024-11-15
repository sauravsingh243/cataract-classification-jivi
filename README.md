# cataract-classification-jivi

## Setup Instructions
1. **Install Requirements:**  
   Ensure you have Python installed. Install the required dependencies using the command:
   ```
   pip install -r requirements.txt
   ```

2. **Configure Parameters:**
The config.ini file includes adjustable parameters such as:

```
epochs, batch_size, patience, port_num and target_size(img_size)
```

3. **Dataset Preparation:**
Download the dataset from Kaggle.
Place the dataset in the dataset folder in the root directory of the project.

## Usage
1. **Model Training:**
To start training your model, use the following command:
```
python executor.py --flag_training "True"
```
2. **Start API Server:**
If the model is already trained, launch the API server by setting training flag to false:
```
python executor.py --flag_training "False"
```

3. **Test on an Image:**
Test your model on a specific image using the get_results.py script:
```
python get_results.py --file_path "path_to_file"
```
Replace "path_to_file" with the absolute or relative path to your test image.


