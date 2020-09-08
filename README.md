# ml-cleavage
Here is the script that is used to train and test several machine learning models.  
Currently, the script supports following machine learning models:
* Logistic Regression  
* Random Forest   
* Decision Tree  
* SVM  
* Neural Network  
## Usage
To train and test, you could use the following command:  
```
cd scripts  
python test.py -m logistic_regression -i A171T -class 3 --C 1 --solver lbfgs --penalty l2
```
Supportive input data format should be dataframe in binary. An example is in *Data* folder. You could use the following command in jupyter notebook to take a look:  
```
import pickle as pkl
X_train = pkl.load(open("X_train_example","rb"))
y_train = pkl.load(open("y_train_example","rb"))
X_test = pkl.load(open("X_test_example","rb"))
y_test = pkl.load(open("y_test_example","rb"))
```
