#*---config:utf-8---------
import heartpy as hp
import matplotlib.pyplot as plt
import numpy as np
data = hp.get_data('A_ANGER_2_0.csv')
working_data, measures = hp.process(data, 1000.0)
np.savetxt(r"C:\Users\akito\Desktop\test\peaklist.csv",working_data["peaklist"],delimiter=',')
np.savetxt(r"C:\Users\akito\Desktop\test\RRI_list.csv",working_data["RR_list"],delimiter=',')
#plt.plot(working_data["peaklist"], working_data["RR_list"])
#plt.show()
#hp.plotter(working_data, measures)

#np.savetxt("result.csv",            # ファイル名
#           X=working_data["peaklist"],                  # 保存したい配列
#           delimiter=","            # 区切り文字
#)
