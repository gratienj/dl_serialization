#!/bin/bash

#///////////////////////////////////////////////////////////////////////////
#setting multiproc to 1 :
#///////////////////////////////////////////////////////////////////////////
#force the 2 default proc to be executed in only 1 proc (master)
export OMP_PROC_BIND=MASTER


#///////////////////////////////////////////////////////////////////////////
# compilation
#///////////////////////////////////////////////////////////////////////////

if [ -d build_mpi ]; then
    rm -rf  build_mpi/*
    cd build_mpi
else
    mkdir build_mpi
    cd build_mpi
fi

cmake -DCMAKE_BUILD_TYPE=Release ..

make -j 4 > compil_log.txt


# créer les dossiers:
mkdir "LR" "MLR" "nonLR"


# MPI:
NP_LIST="1 2 3 4 5 6 7 8"
folder_List="LR MLR nonLR"

#///////////////////////////////////////////////////////////////////////////
# exécution
#///////////////////////////////////////////////////////////////////////////


for NP in ${NP_LIST}; do
  PROC=$(($NP-1))
	mpirun -genv I_MPI_PIN_PROCESSOR_LIST=0-$PROC -np $NP ./benchmark_mpi_pt.exe 403200 "../../models/LR_model.pt" 1 10 >  LR/log_$NP.txt
	mpirun -genv I_MPI_PIN_PROCESSOR_LIST=0-$PROC -np $NP ./benchmark_mpi_pt.exe 403200 "../../models/MLR_model.pt" 2 10 >  MLR/log_$NP.txt
	mpirun -genv I_MPI_PIN_PROCESSOR_LIST=0-$PROC -np $NP ./benchmark_mpi_pt.exe 403200 "../../models/non_LR_model.pt" 1 10 >  nonLR/log_$NP.txt

done;

for folder in ${folder_List}; do
  touch ${folder}/duration_scatter.txt ${folder}/duration_convert_to_tensor.txt ${folder}/duration_predict.txt
  touch ${folder}/duration_convert_to_array.txt ${folder}/duration_gather.txt

  for NP in ${NP_LIST}; do
    # duration_scatter
    line=`grep  "duration_scatter"  ${folder}/log_$NP.txt`
    echo "$NP $line" | tee -a ${folder}/duration_scatter.txt

    # duration_convert_to_tensor
    line=`grep  "duration_convert_to_tensor"  ${folder}/log_$NP.txt`
    echo "$NP $line" | tee -a ${folder}/duration_convert_to_tensor.txt

    # duration_predict
    line=`grep  "duration_predict_*"  ${folder}/log_$NP.txt`
    echo "$line" | tee -a ${folder}/duration_predict.txt

    # duration_convert_to_array
    line=`grep  "duration_convert_to_array"  ${folder}/log_$NP.txt`
    echo "$NP $line" | tee -a ${folder}/duration_convert_to_array.txt

    # duration_gather
    line=`grep  "duration_gather"  ${folder}/log_$NP.txt`
    echo "$NP $line" | tee -a ${folder}/duration_gather.txt

  done;
done;


# script python:
#!/bin/bash

function build_graph {
python3 - <<END
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def get_data(text_file: str) -> None:
    data=pd.DataFrame()
    time=[]
    with open (text_file, "r") as f:
        lines = f.readlines()

        #TODO : change this lable to your first label
        previous_label=1
        for i,line in enumerate(lines):
            line=line.split(" ")

            if len(line) !=5:
                continue

            new_label=float(line[0])


            if new_label == previous_label:
                time.append(float(line[-2]))
            else:
                # set time:
                data[previous_label]=time
                # reset time
                time=[]
                time.append(float(line[-2]))
                # update previous_label
                previous_label=new_label

        data[new_label]=time

        f.close()

    return data

def build_data(build_path) -> tuple:
    files=['/duration_scatter.txt',
           '/duration_gather.txt',
           '/duration_convert_to_tensor.txt',
           '/duration_convert_to_array.txt',
           '/duration_predict.txt'
            ]
    folders=['LR',
             'MLR',
             'nonLR']

    data_all=list()
    data=pd.DataFrame()
    for folder in folders:
        for i, file in enumerate(files):
            path=build_path+folder+file
            if i==0:
                data = get_data(path)
                index=data.shape[0]
            else:
                #get data from file
                df = get_data(path)
                df.index=np.arange(index,df.shape[0]+index)
                data=pd.concat([data,df])
                # new value of index:
                index=index+df.shape[0]
        data_all.append(data)

    return data_all



def build_graph(data: tuple):
    # affichage settings: ----------------------
    # plt.style.use('default')

    SMALL_SIZE = 8
    MEDIUM_SIZE = 10


    plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)     # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)     # legend fontsize
    plt.rc('figure', titlesize=MEDIUM_SIZE)   # fontsize of the figure title


    message1="coût convertion de données"
    message2="temps de réponse par proc"


    names=['LR','MLR','nonLR']
    lables=['scatter','gather','conv_to_tensor','conv_to_array']

    # TODO : we can change the number of calls [0,10]
    fct_call_number_to_plt=10

    #--------------------------------------------


    fig, axes= plt.subplots(3,2,figsize=(20,12))

    for k in range(3):
        for i in range(0,4):
            axes[k][0].plot(data[k].iloc[i,:].to_list(),marker='^', label=f"{lables[i]}")
            axes[k][0].set_title(f"inference CPU MPI : modèle {names[k]} - {message1}")
            if k==2: axes[k][0].set_xlabel("number of procs")
            #else: axes[k][0].axes.xaxis.set_visible(False)
            # axes.set_xscale('log')
            axes[k][0].set_yscale('log')
            axes[k][0].set_ylabel("time (secondes)")

            axes[k][0].grid(True,which="both", linestyle='--')
            axes[k][0].legend()


        for i in range(0,fct_call_number_to_plt):
            axes[k][1].plot(data[k].columns,data[k].iloc[i+4,:].to_list(),marker='^', label=f"appel n°: {i}")
            axes[k][1].set_title(f"inference CPU MPI : modèle {names[k]} - {message2}")
            if k==2: axes[k][1].set_xlabel("number of procs")
            #else: axes[k][1].axes.xaxis.set_visible(False)
            # axes.set_xscale('log')
            axes[k][1].set_yscale('log')
            axes[k][1].set_ylabel("time (secondes)")

            axes[k][1].grid(True,which="both", linestyle='--')
            axes[k][1].legend()



    plt.show()
    #fig.savefig("icons/inference_cpu_scaling_benchmark.png")
    print("fin sauvegarde")


def compute_factor(data_tuple: list) -> list:
    delta=[]
    for data_in in data_tuple:
        row=data_in.shape[0]
        col=data_in.shape[1]-1

        scale=[data_in.columns[j+1]-data_in.columns[j-1] for j in range(1,col)]

        # initialize dicts:
        data={scale[j]:[] for j in range(0,col-1)}


        for i in range(row):
            for j in range(1,col):
                r=(data_in.iloc[i , j+1]-data_in.iloc[i , j-1])/2
                data[scale[j-1]].append(r)

        df_data=pd.DataFrame.from_dict(data)

        delta.append(df_data)

    return delta

# build data frame from data files:
data=build_data('./')

# build scalability  graph:-----
build_graph(data)

END
}

# Call it
build_graph



cd ..