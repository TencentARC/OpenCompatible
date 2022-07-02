# Dataset Preparation

#### Contents:
1. Image Retrieval datasets
   1. [Google Landmark v2 (clean version)](#Google-Landmark-v2)
   2. [Revisited Oxford and Paris](#Revisited-Oxford-and-Paris)
2. Face Recognition datasets:
   1. MS1M-v3
   2. IJB-C
3. Person Re-ID datasets:
   1. ongoing

## Google Landmark v2

1. Download dataset

    Prepare `training set` ("train set"), `query set` ("test set") and `gallery set` ("index set") from [https://github.com/cvdfoundation/google-landmark](https://github.com/cvdfoundation/google-landmark).


2. Prepare indices and ground truth
    
    You can directly use our provided ground truth ([Google Drive]()), or prepare files by yourself:

   - Download indices of training set ([label_81313.txt](https://drive.google.com/file/d/1IH1nMml9GjIrKVloF9AZO06E28jnXP75/view)) from [https://github.com/Raykoooo/3rd_place_to_Kaggle_Google_Landmark_Retrieval_2020](https://github.com/Raykoooo/3rd_place_to_Kaggle_Google_Landmark_Retrieval_2020).
   
     Contents in `label_81313.txt' are like:
     ```
     # img_path                         # catagory
     # train/6/2/d/62d0caa23ab29732.jpg 77574
     ```
     <br />
    
   - Download [index.csv](https://s3.amazonaws.com/google-landmark/metadata/index.csv) into `data/annotation/gldv2/`, and run:
   
     ```python
     python -m utils.process_gldv2_dataset --root_path '../annotation/gldv2'  --gene_gallery
     ```
     Contents in `gldv2_gallery_list.txt' are like:
     ```
     # img_path                         # indices
     # index/f/d/f/fdf40612109ad174.jpg 0
     ```
     <br />
     
   - Download [retrieval_solution_v2.1.csv](https://s3.amazonaws.com/google-landmark/ground_truth/retrieval_solution_v2.1.csv) into `data/gldv2/`, and run:
   
     ```python
     python -m utils.process_gldv2_dataset --root_path '../annotation/gldv2' --gene_query
     ```
     Contents in `gldv2_private_query_list.txt' are like:
     ```
     # img_path                        # indices
     # test/6/7/e/67e40359b5e315cc.jpg 0
     ```
     Contents in `gldv2_private_query_gt.txt' are like:
     ```
     # img_path                        # indices # gallery_indices_gt
     # test/e/8/e/e8e62a3e45c0bfbb.jpg 7         355411,257490,448713,358167,211709,196095,287824,387112
     ```


## Revisited Oxford and Paris

Download `query set` and `gallery set` ([gnd_roxford5k.pkl](http://cmp.felk.cvut.cz/revisitop/data/datasets/roxford5k/gnd_roxford5k.pkl) and [gnd_rparis6k.pkl](http://cmp.felk.cvut.cz/revisitop/data/datasets/rparis6k/gnd_rparis6k.pkl)) into `data/annotation/roxford/`.

   
##### Folder Structure
  ```
  user_path/
  │
  ├── data/ 
  │   ├── gldv2/
  │       ├── train/
  │           └── ...
  │       ├── index/
  │           └── ...
  │       └── test/
  │           └── ...
  │   ├── roxford/
  │       ├── all_souls_000000.jpg
  │       └── ...
  │   └── rparis/
  │       ├── paris_defense_000000.jpg
  │       └── ...
  │
  ├── annotation/ 
  │   ├── gldv2/
  │       ├── label_81313.txt
  │       ├── gldv2_gallery_list.txt
  │       ├── gldv2_private_query_list.txt
  │       └── ...
  │   └── roxford/
  │       ├── roxford.pkl
  │       ├── roxford_gallery_list.txt
  │       └── ...
  └── ...
  ```




