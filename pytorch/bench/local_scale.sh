#!/bin/bash


#///////////////////////////////////////////////////////////////////////////
#setting multiproc to 1 :
#///////////////////////////////////////////////////////////////////////////
#force the 2 default proc to be executed in only 1 proc (master)
export OMP_PROC_BIND=MASTER


#///////////////////////////////////////////////////////////////////////////
# compilation
#///////////////////////////////////////////////////////////////////////////

if [ -d build_scale ]; then
    rm -rf  build_scale/*
    cd build_scale
else
    mkdir build_scale
    cd build_scale
fi

cmake -DCMAKE_BUILD_TYPE=Release ..

make -j 4 > compil_log.txt


# créer les dossiers:
mkdir "LR" "MLR" "nonLR"


#///////////////////////////////////////////////////////////////////////////
# data scale:
#///////////////////////////////////////////////////////////////////////////
#size_list="10 100 1000 10000 100000 1000000 10000000 100000000"
size_list="10 100 1000 10000 100000 1000000 10000000"
folders="LR MLR nonLR"

for size in ${size_list}; do
	 numactl --physcpubind=1 --membind=0 $PWD/benchmark_scale_pt.exe $PY_CPU_MODELS_DIR"/LR_model.pt" $size 1 >  LR/log_LR_${size}.txt
	 numactl --physcpubind=1 --membind=0 $PWD/benchmark_scale_pt.exe $PY_CPU_MODELS_DIR"/MLR_model.pt"  $size 2 >  MLR/log_MLR_${size}.txt
   numactl --physcpubind=1 --membind=0  $PWD/benchmark_scale_pt.exe $PY_CPU_MODELS_DIR"/non_LR_model.pt"  $size 1 >  nonLR/log_nonLR_${size}.txt
done;


for folder in ${folders}; do
    touch ${folder}/log_out_${folder}.txt
    for size in ${size_list}; do
      line=`grep  "time_"  ${folder}/log_${folder}_${size}.txt`
      echo "$line" | tee -a ${folder}/log_out_${folder}.txt
    done;
done;




#///////////////////////////////////////////////////////////////////////////
# script python:
#///////////////////////////////////////////////////////////////////////////

#!/bin/bash python
function build_graph {
python3 - <<END
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def data_benchmark(text_file: str) -> None:
    data=pd.DataFrame()
    time=[]
    with open (text_file, "r") as f:
        lines = f.readlines()

        previous_label=10.0
        for i,line in enumerate(lines):
            line=line.split(" ")

            if len(line) !=5:
                continue

            new_label=np.float64(line[0])


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


def build_data(build_path):
    sufix={"LR":"/LR/log_out_LR.txt","MLR":"/MLR/log_out_MLR.txt","nonLR":"/nonLR/log_out_nonLR.txt"}

    # LR :
    LR=build_path+sufix['LR']
    df_LR = data_benchmark(LR)

    # MLR
    MLR=build_path+sufix['MLR']
    df_MLR = data_benchmark(MLR)

    # nonLR
    nonLR=build_path+sufix['nonLR']
    df_nonLR = data_benchmark(nonLR)

    return df_LR, df_MLR, df_nonLR



def build_graph(data_torch: pd.DataFrame, name: str, derive=False):
    # affichage settings:
    if derive==True:
      message1="dérivée temporelle par nombre d'appels"
      message2="dérivée temporelle par data size"
    else:
      message1=""
      message2=""

    # params:
    fct_call_number_to_plt=10

    plt.style.use('default')

    fig, axes= plt.subplots(1,2,figsize=(20,4))

    for i in range(0,data_torch.shape[1]):
      axes[0].plot(data_torch.iloc[:,i].to_list(),marker='^', label=f"data size : {data_torch.columns[i]}")
      axes[0].set_title(f"inference CPU : modèle {name} {message1}")
      axes[0].set_xlabel("nombre d\'appels du modèle ")
      # axes.set_xscale('log')
      axes[0].set_yscale('log')
      axes[0].set_ylabel("time (secondes)")

      axes[0].grid(True,which="both", linestyle='--')
      axes[0].legend()


    for i in range(0,fct_call_number_to_plt):
      axes[1].plot(data_torch.columns,data_torch.iloc[i,:].to_list(),marker='^', label=f"appel n°: {i}")
      axes[1].set_title(f"inference CPU : modèle {name} {message2}")
      axes[1].set_xlabel("data size")
      # axes.set_xscale('log')
      axes[1].set_yscale('log')
      axes[1].set_xscale('log')
      axes[1].set_ylabel("time (secondes)")

      axes[1].grid(True,which="both", linestyle='--')
      axes[1].legend()

    plt.show()
    #fig.savefig("icons/inference_cpu_scaling_benchmark.png")
    print("fin sauvegarde")


def compute_factor(data_torch: pd.DataFrame):
    row=data_torch.shape[0]
    col=data_torch.shape[1]-1

    scale=[data_torch.columns[j+1]-data_torch.columns[j-1] for j in range(1,col)]

    # initialize dicts:
    data={scale[j]:[] for j in range(0,col-1)}


    for i in range(row):
        for j in range(1,col):
            r=(np.log10(data_torch.iloc[i , j+1])- np.log10(data_torch.iloc[i , j-1]))/2
            data[scale[j-1]].append(r)

    df_data=pd.DataFrame.from_dict(data)
    return df_data

df_LR, df_MLR, df_nonLR=build_data('./')

# scalability LR model:-----
build_graph(df_LR,"LR")
delta_LR=compute_factor(df_LR)
build_graph(delta_LR,"LR",True)

# scalability MLR model:-----
build_graph(df_MLR,"MLR")
delta_MLR=compute_factor(df_MLR)
build_graph(delta_MLR,"MLR",True)

# scalability nonLR model:-----
build_graph(df_nonLR,"nonLR")
delta_nonLR=compute_factor(df_nonLR)
build_graph(delta_nonLR,"nonLR",True)

END
}

# Call it
build_graph



cd ..