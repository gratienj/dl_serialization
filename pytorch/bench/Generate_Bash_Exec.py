#!/usr/bin/env python


import os

class Generate_Bash:
    def __init__(self,cluster='False',name="local_scale",option='scale'):
        self.cluster=cluster
        self.name=name
        self.option=option
        # TODO : specify options:-------------------------------------------------------------------------
        self.SHARED_BENCH_PY_DIR="/home/irsrvshare2/R11/xca_acai/work/kadik/pytorch"
        self.models_saved_dir= "../../CPU/models"
        self.nb_appel=10

        # scale settings:
        self.size_list_scale='''size_list_scale="10 100 1000 10000 100000 1000000 10000000" '''
        self.scale_cpp="benchmark_scale_pt.exe"


        #-------------------------------------------------------------------------------------------------
        self.list_models=[model for model in os.listdir(self.models_saved_dir) if model.endswith('pt')]
        self.folders=[ folder.split('.')[0] for folder in self.list_models]
        self.folders_fPATH=[os.path.join(self.models_saved_dir, folder.split('.')[0]) for folder in self.list_models]

        # generate code:
        self.separator='\n#///////////////////////////////////////////////////////////////////////////'
        self.code="#!/bin/bash"
        self.tab=" "*4
        self.set_up()


    def space(self,s: int):
        sp=""
        for i in range(s):
            sp+="\n"
        self.code+=sp

    def add_title(self,message):
        self.code+=self.separator
        self.code+=f"\n#{message}"
        self.code+=self.separator

    def add_comment(self, message,tab=0):
        self.code+=f"\n{' '*tab}#{message}"


    def add_code(self, code: str, tab: int=0):
        self.code+=f'''\n{" "*tab}{code}'''

    def build_folder(self):
        build='build_'+"GPU"
        self.add_code(f'if [ -d {build} ]; then')
        self.add_code(f'rm -rf  {build}/*',4)
        self.add_code(f'cd {build}',4)
        self.add_code('else')
        self.add_code(f'mkdir {build}',4)
        self.add_code(f'cd {build}',4)
        self.add_code('fi')

    def add_folder(self, folder):
        self.code+=f'''\nmkdir {folder}'''

    def get_feature(self,folder):
        l=list(folder)
        num=''
        for i in range(len(l)-1,-1,-1):
            if l[i].isdecimal():
                num+=l[i]
            else:
                break
        feature=int(num[::-1])
        return feature


    def set_up(self):
        self.space(2)

        if self.cluster==True:
            self.add_title('cluster GPU settings: ')
            self.add_code("#SBATCH --wckey mga88110")
            self.add_code("#SBATCH -J Geoxim_IA_Inference")
            self.add_code("#SBATCH -N 1")
            self.add_code("#SBATCH -n 1")
            self.add_code("#SBATCH -p gpgpu")
            self.add_code("#SBATCH --ntasks-per-node=1")
            self.add_code("#SBATCH --cpus-per-task=36")
            self.add_code("##SBATCH --gpus=1   -- this command is not used with --gres in the update of slurm in 29/07/2022")
            self.add_code("#SBATCH --gres=gpu:1")
            self.add_code("#SBATCH --time=08:00:00")
            self.add_code("#SBATCH --exclusive")
            self.add_code("#SBATCH --verbose")
            #self.add_code("#SBATCH --error=RUN-MPI.err")
            #self.add_code("#SBATCH --output=RUN-MPI.out")
            self.add_code("#SBATCH --switches=1@1:0:0")
            self.space(1)



        # compilation:
        self.add_title("compilation: ")
        self.build_folder()
        self.space(1)
        self.add_code('cmake -DCMAKE_BUILD_TYPE=Release ..')
        self.add_code('make -j 4 > compil_log.txt')
        self.space(2)

        # add folder for each model:
        self.add_title("add folders for models")
        folders_str='''folders=" '''
        folders_list=[]

        for folder in self.folders:
            f=folder.split('.')[0]
            if f not in ['LR_model', 'MLR_model', 'non_LR_model']:
                self.add_folder(f)
                folders_str+=folder+" "
                folders_list.append(f)
            else :
                continue
        folders_str+=''' " '''

        # mise à jour folders:
        self.folders=folders_list

        self.add_code(folders_str)
        self.space(1)

        # execute scale:
        if self.option=='scale':
            # data scale:
            self.add_title('data scale :')
            self.add_code(self.size_list_scale)
            self.space(2)

            self.add_code("for size in ${size_list_scale}; do")
            for folder in self.folders:
                feature=self.get_feature(folder)
                self.add_code(f'''./{self.scale_cpp} $PY_CPU_MODELS_DIR"/{folder}.pt" $size {feature} {self.nb_appel} >  {folder}/log-{folder}-$size.txt''',4)
            self.add_code('done;')
            self.space(1)
            # get data:
            self.add_code("for folder in ${folders}; do")
            self.add_code("touch ${folder}/log-out-${folder}.txt",4)
            self.add_code("for size in ${size_list_scale}; do",4)
            self.add_code('''line=`grep  "time_"  ${folder}/log-${folder}-$size.txt`''',8)
            self.add_code('''echo "$line" | tee -a ${folder}/log-out-${folder}.txt''',8)
            self.add_code("done;",4)
            self.add_code("done;")


        # copy code to shared network zone:
        if self.cluster==True:
            self.add_title('''copy results to shared zone :''')
            shared_path_dir=os.path.join(self.SHARED_BENCH_PY_DIR,"GPU")
            self.add_code(f'''SHARED_BENCH_PY_DIR={shared_path_dir}''')
            self.add_code('''rm -rf ${SHARED_BENCH_PY_DIR}/*''')
            self.add_code('''cp -R ${folders} ${SHARED_BENCH_PY_DIR}''')

        # cd .. :
        self.add_code("cd ..")
        self.space(2)


        # finished:
        self.add_title("finished")

        filename=f"{self.name}.sh"
        #file = open(os.path.join(self.PATH_DIR,filename),'w')
        file = open(filename,'w')
        file.write(self.code)
        file.close()



if __name__=="__main__":
    #local_scale script
    Generate_Bash(cluster=False,name="local_scale",option='scale')
    #cluster_scale script
    Generate_Bash(cluster=True,name="cluster_scale",option='scale')

