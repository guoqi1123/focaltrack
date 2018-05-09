# draw the mean error vs. i and confidence thresholding
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax1 = fig.add_subplot(2,3,1,title="Error per resolution depth")
ax1.plot([0.54881,0.675819,0.54212,0.321148],"-o")
ax1.set_xticklabels(['$i=0$','','$i=1$','','$i=2$','','$i=3$'])
plt.xlabel('$Z^{i,j,k}, \\forall j,k$')
plt.ylabel('Mean err (m)')

ax2 = fig.add_subplot(2,3,3, title="Error for fused depth")
ax2.plot([0.0989643, 0.0972701,0.0648913,0.0321656],"-o")
ax2.set_xticklabels(['$0$','','$0.9$','','$0.99$','','$0.999$'])
plt.xlabel('Confidence threshold')
plt.ylabel('Mean err (m)')
plt.show()