A Pytorch implementation of "Segmentation-Based Deep-Learning Approach for Surface-Defect Detection"
(https://link.springer.com/article/10.1007/s10845-019-01476-x).

# Training:
It's a two stages learning, Fist stage is a Segmentation Network (SegNet), Second Stage is a Decision Netork (DecNet) for classify the defects.

     1.  Segmentation Network (SegNet)
            -->  Make sure the cfg.TRAIN.TRAIN_MODEL in " lib/config/default.py " is "SegNet".
            -->  Run Training.py. Get the "SegNet_model_best.pth".

     2.  Decision Netork (DecNet) 
            -->  Make sure the cfg.TRAIN.TRAIN_MODEL in " lib/config/default.py " is "DecNet".
            -->  Run " Training.py ". Get the "DecNet_model_best.pth".
        
# Evaluation:
     1.  Run " evaluation.py ". 
