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
