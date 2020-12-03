import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import roc_curve, auc
import  csv
import pandas as pd
Path = "CROSS_ROC.csv"
i = 0
Label = []
RPRes = []
CROSS = []
for line in open(Path):
    i += 1
    list = line[0:-1].split(',')
    if(i != 1):
        Label.append(float(list[1]))
        RPRes.append(float(list[3]))
        CROSS.append(float(list[2]))
fpr1, tpr1, threshold1 = roc_curve(Label, CROSS)
roc_auc1 = auc(fpr1, tpr1)

fpr2, tpr2, threshold2 = roc_curve(Label, RPRes)
roc_auc2 = auc(fpr2, tpr2)


font1 = {

'size'   : 15,
}
plt.figure()
lw =4
plt.figure(figsize=(10, 10), dpi = 300)

plt.plot(fpr1, tpr1, color='darkorange',
         lw=lw, label='CROSS curve (AUC = %0.2f)' % roc_auc1) ###假正率为横坐标，真正率为纵坐标做曲线
plt.plot(fpr2, tpr2, color='blue',
         lw=lw, label='RPRes curve (AUC = %0.2f)' % roc_auc2)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.01])
plt.xlabel('1-Specificity', fontsize = 20)
plt.ylabel('Sensitivity', fontsize = 20)
plt.title('The ROC Curve of RPRes and CROSS', fontsize = 20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(loc="lower right", prop = font1)

plt.show()
