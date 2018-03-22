# How to run  

Make sure you have Tensorflow **1.5** installed on GPU.  

The script saves a lot of files (OOF, predictions, models) in the following folders:  
- `temporary_cv_models/`  
- `temporary_cv_preds/`  
- `temporary_oof_preds/`  

Input data should be inside `input/` folder.  

**Make sure to replace the** `embedding` **dictionnary in**`train_nn.py` **with your paths to pre-trained embeddings**   

The *main* file is `train_nn.py`.  
When run, this file asks for a **model type**. This corresponds to models implemented in `models.py`. At the moment, models implemented are:  
- bibigru  
- pooled_gru  
- ngram_cnn  
- cnn_gru  
- gru_cnn  

Feel free to complete `models.py` with your own architecture!  
