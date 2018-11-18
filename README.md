
Code of the paper: Collell, G. & Moens, M-F. (2018) "Do Neural Network Cross-Modal Mappings Really Bridge Modalities?" ACL

Code implemented by Guillem Collell (Department of Computer Science KU Leuven).



** DEPENDENCIES **:

Python > 3.5
tensorflow
 1.4.0
scikit-learn 0.19.1


** 
HOW TO RUN THE CODE **:



# Step 0: Download the data (http://liir.cs.kuleuven.be/software_pages/cross_modal_acl.php). The code already comes with vgg128 image embeddings (as they are compact and take little space) 

# Step 1: Run the cross-modal mapping (which evaluates mNNO) by running (inside the "code" folder): 

python learn_and_eval.py

If you did not download the data in Step 0, you can still run the code with the vgg128 and biGRU embeddings in the IAPR TC-12 dataset by typing: 

python learn_and_eval.py --vis_emb ['vgg128']

To see all possible options (arguments) type:

python learn_and_eval.py --help




