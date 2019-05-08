#*---config:utf-8---------
import heartpy as hp
import matplotlib.pyplot as plt
data = hp.get_data('A_ANGER_2_0.csv')
working_data, measures = hp.process(data, 1000.0)
plt.plot(working_data["peaklist"][0:500], working_data["RR_list"][0:500])
plt.show()
#hp.plotter(working_data, measures)

#np.savetxt("result.csv",            # ファイル名
#           X=working_data["peaklist"],                  # 保存したい配列
#           delimiter=","            # 区切り文字
#)
