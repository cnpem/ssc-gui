import ast
import ipywidgets as widgets 
from ipywidgets import fixed 


field_style = {'description_width': 'initial'}

def update_imshow(sinogram,figure,subplot,frame_number,top=0, bottom=None,left=0,right=None,axis=0,title=False,clear_axis=True,cmap='gray',norm=None):
    subplot.clear()
    if bottom == None or right == None:
        if axis == 0:
            subplot.imshow(sinogram[frame_number,top:bottom,left:right],cmap=cmap,norm=norm)
        elif axis == 1:
            subplot.imshow(sinogram[top:bottom,frame_number,left:right],cmap=cmap,norm=norm)
        elif axis == 2:
            subplot.imshow(sinogram[top:bottom,left:right,frame_number],cmap=cmap,norm=norm)
    else:
        if axis == 0:
            subplot.imshow(sinogram[frame_number,top:-bottom,left:-right],cmap=cmap,norm=norm)
        elif axis == 1:
            subplot.imshow(sinogram[top:-bottom,frame_number,left:-right],cmap=cmap,norm=norm)
        elif axis == 2:
            subplot.imshow(sinogram[top:-bottom,left:-right,frame_number],cmap=cmap,norm=norm)
    if title == True:
        subplot.set_title(f'Frame #{frame_number}')
    if clear_axis == True:
        subplot.set_xticks([])
        subplot.set_yticks([])    
    figure.canvas.draw_idle()

def read_string_as_list(str):
    return ast.literal_eval(str)

class VideoControl:
    def __init__ (self,slider,step,interval,description):
    
        value, minimum, maximum = slider.widget.value,slider.widget.min,slider.widget.max

        self.widget = widgets.Play( value=value,
                                    min=minimum,
                                    max=maximum,
                                    step=step,
                                    interval=interval,
                                    description=description,
                                    disabled=False )

        widgets.jslink((self.widget, 'value'), (slider.widget, 'value'))

class Button:

    def __init__(self,description="DESCRIPTION",layout=widgets.Layout(),icon=""):
        self.button_layout = layout
        self.widget = widgets.Button(description=description,layout=self.button_layout,icon=icon,style=field_style)

    def trigger(self,func):
        self.widget.on_click(func)

class Input(object):

    def __init__(self,type,dictionary,key,field_description="",layout=None,bounded=()):
        
        self.dictionary = dictionary
        self.key = key
        
        if layout == None: 
            self.items_layout = widgets.Layout()
        else:
            self.items_layout = layout
   
        if type == 'Checkbox':
            self.widget = widgets.Checkbox(description=field_description,value=self.dictionary[self.key],layout=self.items_layout, style=field_style)
        elif type == "IntSlider":
            self.widget = widgets.IntSlider(description=field_description,value=self.dictionary[self.key],layout=self.items_layout, style=field_style,min=bounded[0],max=bounded[1],step=bounded[2],)
        elif type == "IntText":
            self.widget = widgets.IntText(description=field_description,value=self.dictionary[self.key],layout=self.items_layout, style=field_style)
        elif type == "BoundedIntText":
            self.widget = widgets.BoundedIntText(description=field_description,value=self.dictionary[self.key],layout=self.items_layout, style=field_style, min=bounded[0],max=bounded[1],step=bounded[2],)
        elif type == "FloatText":
            self.widget = widgets.FloatText(description=field_description,value=self.dictionary[self.key],layout=self.items_layout, style=field_style)
        elif type == "BoundedFloatText":
            self.widget = widgets.BoundedFloatText(description=field_description,value=self.dictionary[self.key],layout=self.items_layout, style=field_style,min=bounded[0],max=bounded[1],step=bounded[2],)
        elif type == "TextString":
            self.widget = widgets.Text(description=field_description,value=self.dictionary[self.key],layout=self.items_layout, style=field_style)
        elif type == "TextList":
            self.widget = widgets.Text(description=field_description,value=str(self.dictionary[self.key]),layout=self.items_layout, style=field_style)
        elif type == "TextDict":
            self.widget = widgets.Text(description=field_description,value=str(self.dictionary[self.key]),layout=self.items_layout, style=field_style)
        
        widgets.interactive_output(self.update_dict_value,{'value':self.widget}) # monitor changes in widget and call function when it occurs

    def update_dict_value(self,value):
        if isinstance(self.dictionary[self.key],list) or isinstance(self.dictionary[self.key],dict):
            if isinstance(value,str):
                self.dictionary[self.key] = read_string_as_list(value)
            else :
                self.dictionary[self.key] = value    
        else:
            self.dictionary[self.key] = value    

def slide_and_play(slider_layout=widgets.Layout(width='90%'),label="",description="",frame_time_milisec = 0):

    def update_frame_time(play_control,time_per_frame):
        play_control.widget.interval = time_per_frame

    selection_slider = Input({"dummy_key":1},"dummy_key",description=description, bounded=(0,100,1),slider=True,layout=widgets.Layout(width='max-width'))
    play_control = VideoControl(selection_slider,1,100,"Play Button")

    pbox = widgets.Box([play_control.widget],layout=get_box_layout('max-width'))

    if frame_time_milisec != 0:
        frame_time = Input({"dummy_key":frame_time_milisec},"dummy_key",description="Time/frame [ms]",layout=widgets.Layout(width='160px'))
        widgets.interactive_output(update_frame_time, {'play_control':fixed(play_control),'time_per_frame':frame_time.widget})
        play_box = widgets.HBox([selection_slider.widget,widgets.Box([pbox,frame_time.widget],layout=get_box_layout('max-width'))])
    else:
        play_box = widgets.HBox([selection_slider.widget, play_control.widget])

    if label != "":
        play_label = widgets.HTML(f'<b><font size=4.9px>{label}</b>' )
        play_box = widgets.VBox([play_label,play_box])

    return play_box, selection_slider,play_control

def get_box_layout(width,flex_flow='column',align_items='center',border='1px none black'):
    return widgets.Layout(flex_flow=flex_flow,align_items=align_items,border=border,width=width)



####### OBSOLETE

# def run_ptycho_from_jupyter(mafalda,python_script_path,json_filepath_path,output_path="",slurm_filepath = 'ptychoJob2.srm',jobName='jobName',queue='cat',GPUs=1,cpus=32):
#     slurm_file = write_slurm_file(python_script_path,json_filepath_path,output_path,slurm_filepath,jobName,queue,GPUs,cpus)
#     call_cmd_terminal(slurm_file,mafalda,remove=False)
    
# def run_ptycho(dummy):
#     pythonScript = global_paths_dict["ptycho_script_path"]
#     json_filepath = global_paths_dict["json_filepath"]
#     slurm_filepath = global_paths_dict["slurm_filepath"]

#     print(f'Running ptycho with {machine_selection.value} machine...')

#     global global_dict

#     from pprint import pprint
#     pprint(global_dict)

#     if machine_selection.value == 'Local':
#         cmd = f'python3 {pythonScript} {json_filepath}'
#         # cmd = f'python3 ~/ssc-cdi/bin/sscptycho_main_test.py {json_filepath}'
#         print('Running command: ',cmd)               
#         output = call_and_read_terminal(cmd,mafalda,use_mafalda=False)
#         print(output.decode("utf-8"))
#     elif machine_selection.value == "Cluster": 
#         global jobNameField, jobQueueField, cpus, GPUs
#         jobName_value = jobNameField.widget.value
#         queue_value   = jobQueueField.widget.value
#         cpus_value    = cpus.widget.value
#         GPUs_value    = GPUs.widget.value
#         create_directory_if_doesnt_exist(global_paths_dict["output_folder"])
#         run_ptycho_from_jupyter(mafalda,pythonScript,json_filepath,output_path=global_paths_dict["output_folder"],slurm_filepath = slurm_filepath,jobName=jobName_value,queue=queue_value,GPUs=GPUs_value,cpus=cpus_value)



# def call_and_read_terminal(cmd,mafalda,use_mafalda=True):
#     if use_mafalda == False:
#         p = Popen(cmd, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True)
#         terminal_output = p.stdout.read() # Read output from terminal

#     else:
#         stdin, stdout, stderr = mafalda.exec_command(cmd)
#         terminal_output = stdout.read() 
#         # print('Output: ',terminal_output)
#         # print('Error:  ',stderr.read())
#     return terminal_output

# def call_cmd_terminal(filename,mafalda,remove=False):
#     cmd = f'sbatch {filename}'
#     terminal_output = call_and_read_terminal(cmd,mafalda).decode("utf-8") 
#     print('Terminal output:',terminal_output)
#     if remove: # Remove file after call
#         cmd = f'rm {filename}'
#         subprocess.call(cmd, shell=True)
        
# def monitor_job_execution(given_jobID,mafalda):
#     sleep_time = 10 # seconds
#     print(f'Starting job #{given_jobID}...')
#     time.sleep(3) # sleep for a few seconds to wait for job to really start
#     jobDone = False
#     job_duration = 0
#     while jobDone == False:
#         time.sleep(sleep_time)
#         job_duration += sleep_time
#         cmd = f'squeue | grep {given_jobID}'
#         terminal_output = call_and_read_terminal(cmd,mafalda).decode("utf-8") 
#         if given_jobID not in terminal_output:
#             jobDone = True
#         else:
#             print(f'\tWaiting for job {given_jobID} to finish. Current duration: {job_duration/60:.2f} minutes')
#     return print(f"\t \t Job {given_jobID} done!")

# def write_slurm_file(python_script_path,json_filepath_path,output_path="",slurm_filepath = 'slurmJob.sh',jobName='jobName',queue='cat',GPUs=1,cpus=32):
#     # Create slurm file
#     logfiles_path = slurm_filepath.rsplit('/',2)[0]
#     string = f"""#!/bin/bash

# #SBATCH -J {jobName}          # Select slurm job name
# #SBATCH -p {queue}            # Fila (partition) a ser utilizada
# #SBATCH --gres=gpu:{GPUs}     # Number of GPUs to use
# #SBATCH --ntasks={cpus}       # Number of CPUs to use. Rule of thumb: 1 GPU for each 32 CPUs
# #SBATCH -o {logfiles_path}/logfiles/{username}_slurm.log        # Select output path of slurm file

# source /etc/profile.d/modules.sh # need this to load the correct python version from modules

# module load python3/3.9.2
# module load cuda/11.2

# python3 {python_script_path} {json_filepath_path} > {os.path.join(logfiles_path,'logfiles',f'{username}_ptycho_output.log')} 2> {os.path.join(logfiles_path,'logfiles',f'{username}_ptycho_error.log')}
# """
    
#     with open(slurm_filepath,'w') as the_file:
#         the_file.write(string)
    
#     return slurm_filepath
