# IDUO.jl
This is a Julia implementation of the Incremental Dictionary Learning with Sparsity by Azimi-Sadjadi, Zhao, Sheedvash

USPS dataset used in testing can be found here: https://www.dropbox.com/sh/4pn9kty1kw8dl67/AADa7YshuXCtW5ouYxcc-Rnpa?dl=0

This dataset has been altered slightly from the original USPS, making it more suitable for this test. In particular, the samples were re-sampled and zero padding was added to the edges of the images so that the format matched MNIST zero-padding spec.

This algorithm, like IK-SVD, is best used for the following scenario: Suppose you have an Sparse Reconstruction based estimation model which was trained on data Y_old and which represents the data well at the chosen sparsity factor i.e. Y ~= DX where X is a matrix of sparse code vectors with each column of coefficients corresponding to the linear combination of atoms (columns) in D that best approximate the corresponding column of Y. Now, suppose we obtain a limited number of labeled/response-verified samples by one process or another giving us new data Y_new, we would like to update our various dictionaries to include this new information without a complete re-training of our dictionaries. IK-SVD and IDUO address this problem by augmenting the dictionaries with K1 new atoms that are learned in a convention similar to K-SVD learning but with a Entropy of Information based criteria for initial atom selection from training samples.

IDUO addresses this problem slightly differently from IK-SVD. The main difference with this method is that in IDUO, the atoms are solved one at a time and never changed after they have been solved wheras IK-SVD alternates between solving sparse codes and then re-solving all atoms. IDUO also utilizes a power method approach to estiamting the atoms whereas IK-SVD utilizes the SVD operation. 
