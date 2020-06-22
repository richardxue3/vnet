from keras.models import load_model
from data import *
from model import *
import matplotlib.pyplot as plt

model = load_model('./Model/Vnet.h5')

# Evaluation in lits datasets
generator = generator('lits_val.h5', train_batch_size = 4, test_batch_size = 1)

(test_data,test_label) = generator.eval(8)  # eval(i) represents the evaluation of  the i-th sample

results = model.predict(test_data, 1, verbose=1)
plt.figure(1)
plt.subplot(131)
plt.title('Image',fontsize=12,color='r')
plt.imshow(test_data[0, :, :, 100, 0])
plt.subplot(132)
plt.title('Label',fontsize=12,color='r')
plt.imshow(test_label[0, :, :, 100, 0])
plt.subplot(133)
plt.title('Predict',fontsize=12,color='r')
plt.imshow(results[0, :, :, 100, 0])
plt.show()

# # Evaluation in sliver datasets
# generator = sliver_generator('E:/wp/Recursive-Cascaded-Networks-master/datasets/sliver_val.h5', train_batch_size = 4, test_batch_size = 1)
# test_data = generator.testgenerator()
# loss, accuracy = model.evaluate_generator(test_data,20,verbose=1)
# print("Accuracy = ", accuracy)
# results = model.predict_generator(test_data,20,verbose=1)