# Contrastive Box Embedding for Collaborative Reasoning

This repository includes the implementation for Contrastive Box Embedding for Collaborative Reasoning (CBox4CR):
*Tingting Liang, Yuanqing Zhang, Qianhui Di, Congying Xia, Youhuizi Li, Yuyu yin. 2023.*

## Requirements
```
pandas==1.3.4
scipy==1.7.1
torch==1.11.0
numpy==1.21.2
Bottleneck==1.3.2
einops==0.4.1
setuptools==58.0.4
```

## Datasets

The processed datasets are in  [`./datasets/`](https://github.com/crazyfat/Cbox4cr/tree/main/datasets/)

**MovieLens Datasets**: The origin dataset can be found [here](https://grouplens.org/datasets/movielens/). 

**Amazon Datasets**: The origin dataset can be found [here](http://jmcauley.ucsd.edu/data/amazon/). 

    

## Example to run the codes

Some running commands can be found in [`./commands/commands.txt`](https://github.com/crazyfat/Cbox4cr/tree/main/commands/)

```
# CBox4CR for recommendation on ML-100k dataset
> cd CBox4CR
> python main.py --dataset movielens_100k --save_load_path saved-models/best_movielens_100k.json --threshold 4 --leave_n 1 --keep_n 5 --max_history_length 5 --n_neg_train 1 --n_neg_val_test 100 --training_batch_size 128 --val_test_batch_size 256 --emb_size 64 --lr 0.0005  --tau 0.01 --theta 0.02 --beta 0.4 --activation 'relu' --cen 0.02 --gamma 12.0 --epsilon 2.0 --num_layers 1 --std_vol -2 --u2i True
```
## Cite
Please consider cite our paper if you find the paper and the code useful.
```
@inproceedings{liang2023contrastive,
  title={Contrastive Box Embedding for Collaborative Reasoning},
  author={Liang, Tingting and Zhang, Yuanqing and Di, Qianhui and Xia, Congying and Li, Youhuizi and Yin, Yuyu},
  booktitle={Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages={38--47},
  year={2023}
}
```
For inquiries contact Yuanqing Zhang (zhangyuanqing@hdu.edu.cn) or Tingting Liang (liangtt@hdu.edu.cn)



