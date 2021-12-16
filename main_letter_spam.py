'''Main function for UCI letter and spam datasets.
'''

# Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np

from data_loader import data_loader
from gain import gain, pmu_gain, pmu1_gain, pmu2_gain, pmu3_gain, pmu4_gain
from utils import rmse_loss
import sys
from matplotlib import pyplot
from cluster import cluster, impute_cluster
import xlsxwriter

test_size = -2600
start_range = 500
end_range = 700

def writexl(data_fixed):
  workbook = xlsxwriter.Workbook('data_fixed2.xlsx')
  worksheet = workbook.add_worksheet()
  for i in range(np.size(data_fixed,0)):
    for j in range(np.size(data_fixed,1)):
      worksheet.write(i,j,data_fixed[i][j])

  workbook.close()

def plotter (test_data, imputed_data, cluster_data, data_m, col, rows, rowe, err=0):
  # test_data = (1-data_m) * test_data
  t = (rowe-rows)/60
  x_axis = np.arange(0.,t,1/60)
  labels = ['Frequency', 'Dropoff', 'Voltage Magnitude', 'Voltage Angle', 'Current Magnitude', 'Current Angle']
  tdata = []
  adata = []
  idata = []
  cdata = []
  ddata = []
  # print (test_data[rows:rowe,54])
  for i in range(rows,rowe):
    # print (test_data[i][col])
    if (test_data[i][col]!=0):
      tdata.append(test_data[i][col])
      if (err<1 and err>=0): 
        adata.append(test_data[i][col] + err*test_data[i][col]) #step attack/ simple manipulation atatck
      else:
        adata.append(test_data[i-err][col]) #replay attack
      idata.append(imputed_data[i][col])
      cdata.append(cluster_data[i][col])
      ddata.append(test_data[i][col+12])

  # print (tdata)
  lw = 3.0
  pyplot.plot(x_axis,tdata, linestyle=':',linewidth=lw)
  # pyplot.plot(x_axis,adata,linewidth=lw)
  pyplot.plot(x_axis,ddata,linewidth=lw)
  pyplot.plot(x_axis,idata,linewidth=lw)
  # pyplot.plot(,linestyle=':')
  fs=14
  pyplot.ylabel(labels[col%6],fontsize=fs)
  pyplot.xlabel('Time',fontsize=fs)
  # pyplot.xlabel('Data point')
  pyplot.legend(['Actual Data', 'Altered Data on PMU5', 'GAIN imputation'], loc='best',fontsize=12) #, 'K-Means Imputation', 
  pyplot.show()
  return adata,idata

def plotter1 (test_data, imputed_data, cluster_data, data_m, col, rows, rowe, err=0):
  # test_data = (1-data_m) * test_data
  t = (rowe-rows)/60
  x_axis = np.arange(0.,t,1/60)
  labels = ['Frequency', 'Dropoff', 'Voltage Magnitude', 'Voltage Angle', 'Current Magnitude', 'Current Angle']
  tdata = []
  adata = []
  idata = []
  cdata = []
  ddata = []
  # print (test_data[rows:rowe,54])
  for i in range(rows,rowe):
    # print (test_data[i][col])
    if (test_data[i][col]!=0):
      tdata.append(test_data[i][col])
      if (err<1 and err>=0): 
        adata.append(test_data[i][col] + err*test_data[i][col]) #step attack/ simple manipulation atatck
      else:
        adata.append(test_data[i-err][col]) #replay attack
      idata.append(imputed_data[i][col])
      cdata.append(cluster_data[i][col])
      # ddata.append(test_data[i][col+12])

  # print (tdata)
  lw = 3.0
  pyplot.plot(x_axis,tdata, linestyle=':',linewidth=lw)
  # pyplot.plot(x_axis,adata)
  # pyplot.plot(x_axis,ddata,linewidth=lw)
  pyplot.plot(x_axis,idata,linewidth=lw)
  # pyplot.plot(,linestyle=':')
  fs=14
  pyplot.ylabel(labels[col%6],fontsize=fs)
  pyplot.xlabel('Time',fontsize=fs)
  # pyplot.xlabel('Data point')
  pyplot.legend(['Actual Data', 'GAIN imputation'], loc='best',fontsize=12) #, 'K-Means Imputation', 
  pyplot.show()
  return adata,idata

def main (args):
  '''Main function for UCI letter and spam datasets.
  
  Args:
    - data_name: letter or spam
    - miss_rate: probability of missing components
    - batch:size: batch size
    - hint_rate: hint rate
    - alpha: hyperparameter
    - iterations: iterations
    
  Returns:
    - imputed_data_x: imputed data
    - rmse: Root Mean Squared Error
  '''
  
  data_name = args.data_name
  miss_rate = args.miss_rate
  
  gain_parameters = {'batch_size': args.batch_size,
                     'hint_rate': args.hint_rate,
                     'alpha': args.alpha,
                     'iterations': args.iterations}
  
  # Load data and introduce missingness
  ori_data_x, miss_data_x, data_m = data_loader(data_name, miss_rate)

  # print (ori_data_x)
  # print (ori_data_x.shape)

  C = cluster(ori_data_x)
  
  # Impute missing data
  if (data_name == 'pmu'):
    imputed_data_x, imputed_data_test = pmu_gain(miss_data_x, gain_parameters)
    # final_data = np.concatenate((ori_data,imputed_data), axis=1)
    ori_data = ori_data_x[:test_size,:]
    test_data = ori_data_x[test_size:,:]
    data_m_test = data_m[test_size:,:]

    diff_data_x = (imputed_data_test - test_data)
    np.set_printoptions(threshold=sys.maxsize, suppress=True, precision = 5) #, formatter={'float_kind':'{:f}'.format}
    diff_data = (100*(diff_data_x/test_data))

    print ((1-data_m_test) * diff_data)
    # np.set_printoptions(threshold=sys.maxsize, suppress=True)
    # print ((1-data_m) * imputed_data_x)
    
    # Report the RMSE performance
    print()
    rmse = rmse_loss (ori_data, imputed_data_x, data_m[:test_size,:])
    rmse1 = rmse_loss (test_data, imputed_data_test, data_m_test)
    
    print()
    print('RMSE Performance Train: ' + str(np.round(rmse, 4)))
    print('RMSE Performance Test: ' + str(np.round(rmse1, 4)))

  elif (data_name == 'pmu1'):
    imputed_data_x, imputed_data_test, data_m_test = pmu1_gain(miss_data_x, gain_parameters, ori_data_x, miss_rate)

    test_data = ori_data_x[:-1*test_size,:]

    diff_data_x = (imputed_data_test - test_data)
    np.set_printoptions(threshold=sys.maxsize, suppress=True) #, formatter={'float_kind':'{:f}'.format}
    diff_data = (100*(diff_data_x/test_data))

    print ((1-data_m_test) * diff_data)
    # print ((1-data_m_test) * imputed_data_test)
    # np.set_printoptions(threshold=sys.maxsize, suppress=True)
    # print ((1-data_m) * imputed_data_x)
    
    # Report the RMSE performance
    print()
    rmse = rmse_loss (ori_data_x, imputed_data_x, data_m)
    rmse1 = rmse_loss (test_data, imputed_data_test, data_m_test)
    
    print()
    print('RMSE Performance Train: ' + str(np.round(rmse, 4)))
    print('RMSE Performance Test: ' + str(np.round(rmse1, 4)))

  elif (data_name == 'pmu2'):
    imputed_data_x, imputed_data_test, data_m_test = pmu2_gain(miss_data_x, gain_parameters, ori_data_x, miss_rate)

    test_data = ori_data_x[:-1*test_size,:]

    test_data1 = test_data.copy()
    test_data1[data_m_test == 0] = np.nan

    cluster_fixed = impute_cluster(C,test_data1)

    # diff_data_x = (imputed_data_test - test_data)
    # np.set_printoptions(threshold=sys.maxsize, suppress=True) #, formatter={'float_kind':'{:f}'.format}
    # diff_data = (100*(diff_data_x/test_data))

    # writexl ((1-data_m_test) * diff_data)
    # print ((1-data_m_test) * imputed_data_test)
    # np.set_printoptions(threshold=sys.maxsize, suppress=True)
    # print ((1-data_m_test) * cluster_fixed)
    
    # Report the RMSE performance
    # print()
    rmse = rmse_loss (ori_data_x, imputed_data_x, data_m)
    rmse1 = rmse_loss (test_data, imputed_data_test, data_m_test)
    rmse2 = rmse_loss (test_data, cluster_fixed, data_m_test)
    
    print()
    print('RMSE Performance Train: ' + str(np.round(rmse, 4)))
    print('RMSE Performance Test: ' + str(np.round(rmse1, 4)))
    print('RMSE Performance Cluster: ' + str(np.round(rmse2, 4)))

    adata0,idata0 = plotter(test_data, imputed_data_test, cluster_fixed, data_m_test, 0, 1400,1500)
    adata1,idata1 = plotter(test_data, imputed_data_test, cluster_fixed, data_m_test, 1, 1400,1500,0.02)
    adata2,idata2 = plotter(test_data, imputed_data_test, cluster_fixed, data_m_test, 2, 1400,1500,0.01)

    perc1Y = [0]*len(adata0)
    for i in range(len(adata0)):
      perc1Y[i] = float(100*(idata0[i] - adata0[i])/idata0[i])
    perc3Y = [0]*len(adata1)
    for i in range(len(adata1)):
      perc3Y[i] = float(100*(idata1[i] - adata1[i])/idata1[i])
    perc2Y = [0]*len(adata2)
    for i in range(len(adata2)):
      perc2Y[i] = float(100*(idata2[i] - adata2[i])/idata2[i])

    pyplot.plot(perc1Y)
    pyplot.plot(perc3Y)
    pyplot.plot(perc2Y)
    # pyplot.plot(,linestyle=':')
    pyplot.ylabel('Percentage Error')
    # pyplot.xlabel('Data point')
    pyplot.legend(['Frequency','Voltage Magnitude', 'Voltage Angle'], loc='upper right')
    pyplot.show()

  elif (data_name == 'pmu3'):
    imputed_data_x, imputed_data_test, data_m_test = pmu3_gain(miss_data_x, gain_parameters, ori_data_x, miss_rate)

    # print (imputed_data_test[:,60:])
    test_data = ori_data_x[:-1*test_size,:]

    test_data1 = test_data.copy()
    test_data1[data_m_test == 0] = np.nan

    # print (test_data1)

    cluster_fixed = impute_cluster(C,test_data1)

    # diff_data_x = (imputed_data_test - test_data)
    # np.set_printoptions(threshold=sys.maxsize, suppress=True) #, formatter={'float_kind':'{:f}'.format}
    # diff_data = (100*(diff_data_x/test_data))

    # writexl ((1-data_m_test) * diff_data)
    # print ((1-data_m_test) * imputed_data_test)
    # np.set_printoptions(threshold=sys.maxsize, suppress=True)
    # print ((1-data_m_test) * cluster_fixed)
    
    # Report the RMSE performance
    # print()
    rmse = rmse_loss (ori_data_x, imputed_data_x, data_m)
    rmse1 = rmse_loss (test_data, imputed_data_test, data_m_test)
    rmse2 = rmse_loss (test_data, cluster_fixed, data_m_test)
    
    print()
    print('RMSE Performance Train: ' + str(np.round(rmse, 4)))
    print('RMSE Performance Test: ' + str(np.round(rmse1, 4)))
    print('RMSE Performance Cluster: ' + str(np.round(rmse2, 4)))

    err = 0.0 #acc to attack

    adata0,idata0 = plotter(test_data, imputed_data_test, cluster_fixed, data_m_test, 0, start_range,end_range,err)
    # adata1,idata1 = plotter(test_data, imputed_data_test, cluster_fixed, data_m_test, 55, start_range,end_range,err)
    adata2,idata2 = plotter(test_data, imputed_data_test, cluster_fixed, data_m_test, 2, start_range,end_range,err)
    adata3,idata3 = plotter(test_data, imputed_data_test, cluster_fixed, data_m_test, 3, start_range,end_range,err)
    err = 0.05 #acc to attack

    # adata0,idata0 = plotter(test_data, imputed_data_test, cluster_fixed, data_m_test, 0, start_range,end_range,err)
    # adata1,idata1 = plotter(test_data, imputed_data_test, cluster_fixed, data_m_test, 55, start_range,end_range,err)
    # adata2,idata2 = plotter(test_data, imputed_data_test, cluster_fixed, data_m_test, 2, start_range,end_range,err)
    # adata3,idata3 = plotter(test_data, imputed_data_test, cluster_fixed, data_m_test, 3, start_range,end_range,err)
    err = 0.02 #acc to attack

    # adata0,idata0 = plotter(test_data, imputed_data_test, cluster_fixed, data_m_test, 0, start_range,end_range,err)
    # # adata1,idata1 = plotter(test_data, imputed_data_test, cluster_fixed, data_m_test, 55, start_range,end_range,err)
    # adata2,idata2 = plotter(test_data, imputed_data_test, cluster_fixed, data_m_test, 2, start_range,end_range,err)
    # adata3,idata3 = plotter(test_data, imputed_data_test, cluster_fixed, data_m_test, 3, start_range,end_range,err)
    err = 0 #acc to attack

    # adata0,idata0 = plotter(test_data, imputed_data_test, cluster_fixed, data_m_test, 0, 500,700,err)
    # # adata1,idata1 = plotter(test_data, imputed_data_test, cluster_fixed, data_m_test, 55, 500,700,err)
    # adata2,idata2 = plotter(test_data, imputed_data_test, cluster_fixed, data_m_test, 2, 500,700,err)
    # adata3,idata3 = plotter(test_data, imputed_data_test, cluster_fixed, data_m_test, 3, 500,700,err)
    err = 0 #acc to attack

    adata0,idata0 = plotter1(test_data, imputed_data_test, cluster_fixed, data_m_test, 0, start_range,end_range,err)
    # adata1,idata1 = plotter(test_data, imputed_data_test, cluster_fixed, data_m_test, 55, start_range,end_range,err)
    adata2,idata2 = plotter1(test_data, imputed_data_test, cluster_fixed, data_m_test, 2, start_range,end_range,err)
    adata3,idata3 = plotter1(test_data, imputed_data_test, cluster_fixed, data_m_test, 3, start_range,end_range,err)


    perc0Y = [0]*len(adata0)
    for i in range(len(adata0)):
      perc0Y[i] = float(100*(idata0[i] - adata0[i])/idata0[i])
    # perc1Y = [0]*len(adata1)
    # for i in range(len(adata1)):
    #   perc1Y[i] = float(100*(idata1[i] - adata1[i])/idata1[i])
    perc2Y = [0]*len(adata2)
    for i in range(len(adata2)):
      perc2Y[i] = float(100*(idata2[i] - adata2[i])/idata2[i])
    perc3Y = [0]*len(adata3)
    for i in range(len(adata3)):
      perc3Y[i] = float(100*(idata3[i] - adata3[i])/idata3[i])
    # perc4Y = [0]*len(adata4)
    # for i in range(len(adata4)):
    #   perc4Y[i] = float(100*(idata4[i] - adata4[i])/idata4[i])
    # perc5Y = [0]*len(adata5)
    # for i in range(len(adata5)):
    #   perc5Y[i] = float(100*(idata5[i] - adata5[i])/idata5[i])
    
    # pyplot.plot(perc0Y)
    # # pyplot.plot(perc1Y)
    # pyplot.plot(perc2Y)
    # pyplot.plot(perc3Y)
    # # pyplot.plot(perc4Y)
    # # pyplot.plot(perc5Y)
    # # pyplot.plot(,linestyle=':')
    # pyplot.ylabel('Percentage Error')
    # # pyplot.xlabel('Data point')
    # pyplot.legend(['Frequency', 'Voltage Magnitude', 'Voltage Angle'], loc='upper right')
    # pyplot.show()

  elif (data_name == 'pmu4'):
    imputed_data_x, imputed_data_test, data_m_test = pmu4_gain(miss_data_x, gain_parameters, ori_data_x, miss_rate)

    test_data = ori_data_x[:-1*test_size,:]

    diff_data_x = (imputed_data_test - test_data)
    np.set_printoptions(threshold=sys.maxsize, suppress=True) #, formatter={'float_kind':'{:f}'.format}
    diff_data = (100*(diff_data_x/test_data))

    print ((1-data_m_test) * diff_data)
    # print ((1-data_m_test) * imputed_data_test)
    # np.set_printoptions(threshold=sys.maxsize, suppress=True)
    # print ((1-data_m) * imputed_data_x)
    
    # Report the RMSE performance
    print()
    rmse = rmse_loss (ori_data_x, imputed_data_x, data_m)
    rmse1 = rmse_loss (test_data, imputed_data_test, data_m_test)
    
    print()
    print('RMSE Performance Train: ' + str(np.round(rmse, 4)))
    print('RMSE Performance Test: ' + str(np.round(rmse1, 4)))

  else:
    imputed_data_x = gain(miss_data_x, gain_parameters)

    rmse = rmse_loss (ori_data_x, imputed_data_x, data_m)
    print()
    print('RMSE Performance Train: ' + str(np.round(rmse, 4)))
  
  return imputed_data_x, rmse

if __name__ == '__main__':  
  
  # Inputs for the main function
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_name',
      choices=['letter','spam','pmu','pmu1','pmu2','pmu3','pmu4'],
      default='spam',
      type=str)
  parser.add_argument(
      '--miss_rate',
      help='missing data probability',
      default=0.2,
      type=float)
  parser.add_argument(
      '--batch_size',
      help='the number of samples in mini-batch',
      default=128,
      type=int)
  parser.add_argument(
      '--hint_rate',
      help='hint probability',
      default=0.9,
      type=float)
  parser.add_argument(
      '--alpha',
      help='hyperparameter',
      default=100,
      type=float)
  parser.add_argument(
      '--iterations',
      help='number of training interations',
      default=10000,
      type=int)
  
  args = parser.parse_args() 
  
  # Calls main function  
  main(args)
