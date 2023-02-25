A Novel Octree-Based 3-D Fully Convolutional Neural Network for Point Cloud Classification in Road Environment笔记







损失函数

*Loss Function*

For the point cloud classification task in road environments, the main difficulty is that the number of training data per

category is very unbalanced (from tens of thousands of points for the category “electrical wires” to several millions for the category “ground”). Several previous works [47], [50] used a balanced random sampling approach for different categories of the training data. To sum up, they randomly picked the equal number of seed points for each category as the centers of the voxelized areas. The experimental results proved that despite the extremely unbalanced distribution of points from different categories in the original data, balancing the training data among categories obviously enhanced the performance of classification. It can also reduce the running time and overcome the storage difficulties by discarding some training samples when the training data set is too large. However, randomly balanced sampling will lose some valuable potential information that is vital for the construction of CNNs. As a result, the obtained results are inaccurate on the actual test data set.

We alternatively use a simpler strategy to automatically balance the loss among all categories. Specifically, we calculate the weight value for each category by recording its frequency of appearing in all voxels of the octree. On the basis of common cross-entropy loss, we add the weight value for each category to offset the imbalance between all categories.

We define the weighted cross-entropy loss function in the following:

![image-20230216000020946](C:\Users\Liu Dunhui\AppData\Roaming\Typora\typora-user-images\image-20230216000020946.png)



![image-20230216000035111](C:\Users\Liu Dunhui\AppData\Roaming\Typora\typora-user-images\image-20230216000035111.png)

![image-20230216000047953](C:\Users\Liu Dunhui\AppData\Roaming\Typora\typora-user-images\image-20230216000047953.png)



