import os, json
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import getpass

from skimage.restoration import unwrap_phase

import ipywidgets as widgets 
from ipywidgets import fixed

from .misc import Button, Input, update_imshow, slide_and_play


############################################  PATH DEFINITIONS ###########################################################################

username = getpass.getuser()

inputs_folder = "./example/inputs/"
output_folder = "./example/outputs/"

template_dict_path = os.path.join(inputs_folder,'template.json')
output_dict_path = os.path.join(output_folder, f'{username}_input_dict.json') 

DP_filepath = os.path.join(inputs_folder,'example_single_data.npy')
sinogram_filepath = os.path.join(inputs_folder,'complex_sinogram.npy')
probe_filepath = os.path.join(inputs_folder,'example_probes.npy')

global_dict = json.load(open(template_dict_path)) # load from template

############################################  GLOBAL LAYOUTS  ###########################################################################

""" Standard styling definitions """
standard_border = '1px none black'
vbar  = widgets.HTML(value="""<div style="border-left:2px solid #000;height:500px"></div>""")
vbar2 = widgets.HTML(value="""<div style="border-left:2px solid #000;height:1000px"></div>""")
hbar  = widgets.HTML(value="""<hr class="solid" 2px #000>""")
hbar2 = widgets.HTML(value="""<hr class="solid" 2px #000>""")
slider_layout      = widgets.Layout(width='90%',border=standard_border)
slider_layout2     = widgets.Layout(width='30%',flex='flex-grow',border=standard_border)
slider_layout3     = widgets.Layout(display='flex', flex_flow='row',  align_items='flex-start', width='70%',border=standard_border)
items_layout       = widgets.Layout( width='90%',border=standard_border)     # override the default width of the button to 'auto' to let the button grow
items_layout2      = widgets.Layout( width='50%',border=standard_border)     # override the default width of the button to 'auto' to let the button grow
checkbox_layout    = widgets.Layout( width='150px',border=standard_border)   # override the default width of the button to 'auto' to let the button grow
buttons_layout     = widgets.Layout( width='90%',height="40px") # override the default width of the button to 'auto' to let the button grow
center_all_layout  = widgets.Layout(align_items='center',width='100%',border=standard_border) #align_content='center',justify_content='center'
box_layout         = widgets.Layout(flex_flow='column',align_items='flex-start',border=standard_border,width='100%')
sliders_box_layout = widgets.Layout(flex_flow='column',align_items='flex-start',border=standard_border,width='100%')
style = {'description_width': 'initial'}

def get_box_layout(width,flex_flow='column',align_items='flex-start',border=standard_border):
    return widgets.Layout(flex_flow=flex_flow,align_items=align_items,border=border,width=width)


############################################ INTERFACE / GUI : FUNCTIONS ###########################################################################

def create_label_widget(text):
    label = widgets.HTML(value=f"<b style='color:#00008B;font-size:18px;'>{text}</b>")
    return label

def save_on_click(dummy,output_dict_filepath="",dictionary={}):
    print('Saving input json file at: ',output_dict_filepath)
    file = open(output_dict_filepath,"w")
    file.write(json.dumps(dictionary,indent=3,sort_keys=True))
    file.close()
    print('\t Saved!')

############################################ INTERFACE / GUI : TABS ###########################################################################

def inputs_tab():

    global global_dict

    """ Create widgets """
    save_on_click_partial = partial(save_on_click,output_dict_filepath=output_dict_path,dictionary=global_dict)
    saveJsonButton = Button(description="Save Inputs",layout=buttons_layout,icon='fa-floppy-o')
    saveJsonButton.trigger(save_on_click_partial)

    label0 = create_label_widget("Machine Parameters")
    gpus = Input("BoundedIntText",global_dict,'GPUs',bounded=(1,6,1), description="# of GPUs:")
    cpus = Input("BoundedIntText",global_dict,'CPUs',bounded=(1,32,1),description="# of CPUs:")
    box = widgets.HBox([gpus.widget,cpus.widget])

    label1 = create_label_widget("Data Selection")
    data_folder_str = Input('TextString',global_dict,"data_path",  description="Diffraction datapath",layout=items_layout2)
    object_path_str = Input('TextString',global_dict,"object_path",description="Object datapath",layout=items_layout2)
    probe_path_str  = Input('TextString',global_dict,"probe_path" ,description="Probe datapath",layout=items_layout2)
    
    label2 = create_label_widget("Diffraction Pattern")
    global center_y, center_x
    center_x    = Input('IntSlider',{'dummy-key':global_dict["DP_center"][1]},'dummy-key',bounded=(0,50,1),description="Center column (x)",layout=slider_layout)
    center_y    = Input('IntSlider',{'dummy-key':global_dict["DP_center"][0]},'dummy-key',bounded=(0,50,1),description="Center row    (y)",layout=slider_layout)

    label3 = create_label_widget("Post-processing")
    phase_unwrap      = Input('Checkbox',{'dummy-key':global_dict["phase_unwrap"]},'dummy-key',description="Phase Unwrap",layout=checkbox_layout)
    phase_unwrap_box = widgets.Box([phase_unwrap.widget],layout=items_layout2)
    global top_crop, bottom_crop,left_crop,right_crop # variables are reused in crop tab
    top_crop      = Input('IntSlider',{'dummy_key':0},'dummy_key',bounded=(0,10,1), description="Top crop", layout=slider_layout)
    bottom_crop   = Input('IntSlider',{'dummy_key':1},'dummy_key',bounded=(1,10,1), description="Bottom crop", layout=slider_layout)
    left_crop     = Input('IntSlider',{'dummy_key':0},'dummy_key',bounded=(0,10,1), description="Left crop", layout=slider_layout)
    right_crop    = Input('IntSlider',{'dummy_key':1},'dummy_key',bounded=(1,10,1), description="Right crop", layout=slider_layout)

    def update_global_dict(data_folder_str,center_y,center_x,phase_unwrap,top_crop,bottom_crop,left_crop,right_crop):
        global global_dict
        global_dict["data_folder"]  = data_folder_str
        global_dict["DP_center"]    = [center_y,center_x]
        global_dict["phase_unwrap"] = phase_unwrap
        global_dict["crop"] = [top_crop, bottom_crop, left_crop, right_crop]

    """ Monitor variable and call function when they change """
    widgets.interactive_output(update_global_dict,{'data_folder_str':data_folder_str.widget,
                                                    'center_y':center_y.widget,
                                                    'center_x':center_x.widget,
                                                    'phase_unwrap': phase_unwrap.widget,
                                                    'top_crop': top_crop.widget,
                                                    'bottom_crop': bottom_crop.widget,
                                                    'left_crop': left_crop.widget,
                                                    'right_crop': right_crop.widget,
                                                     })

    box = widgets.Box([saveJsonButton.widget,label0,gpus.widget,cpus.widget,label1,data_folder_str.widget,object_path_str.widget,probe_path_str.widget,label2,center_y.widget,center_x.widget,label3,phase_unwrap_box,top_crop.widget,bottom_crop.widget,left_crop.widget,right_crop.widget],layout=box_layout)

    return box

def center_tab():

    output = widgets.Output()
    with output:
        figure, subplot = plt.subplots(figsize=(5,5),constrained_layout=True)
        figure,subplot.imshow(np.random.random((4,4)))
        subplot.set_title('Diffraction Pattern')
        figure.canvas.header_visible = False 
        plt.show()


    def plotshow(figure,subplot,image,title="",figsize=(8,8),savepath=None,show=False):
        subplot.clear()
        from matplotlib.colors import LogNorm
        handle = subplot.imshow(image, interpolation='nearest', cmap = 'viridis', norm=LogNorm())
        if title != "":
            subplot.set_title(title)
        if show:
            if __name__ == '__main__': 
                plt.show()

        figure.canvas.draw_idle()

    def update_mask(figure, subplot,image):
        plotshow(figure,subplot,image)

    def load_difpad(dummy):
        image = np.load(DP_filepath)
        widgets.interactive_output(update_mask,{'figure':fixed(figure), 'subplot': fixed(subplot),'image':fixed(image)})

    load_difpad_button  = Button(description="Load Diffraction Pattern",layout=buttons_layout,icon='folder-open-o')
    load_difpad_button.trigger(load_difpad)

    """ Difpad center boxes """
    sliders_box = widgets.HBox([center_y.widget,center_x.widget],layout=box_layout)
    controls = widgets.Box([load_difpad_button.widget,sliders_box],layout=get_box_layout('500px'))
    box = widgets.HBox([controls,vbar,output])
    return box

def reconstruction_tab():
    
    initial_image = np.random.random((5,5)) # dummy

    output = widgets.Output()
    with output:
        figure, subplot = plt.subplots(figsize=(4,4))
        subplot.imshow(initial_image,cmap='gray')
        figure.canvas.header_visible = False 
        plt.show()


    output3 = widgets.Output()
    with output3:
        figure3, subplot3 = plt.subplots(figsize=(4,4))
        subplot3.imshow(initial_image,cmap='gray')
        figure3.canvas.header_visible = False 
        plt.show()


    output2 = widgets.Output()
    with output2:
        figure2, subplot2 = plt.subplots(figsize=(4,4))
        subplot2.imshow(initial_image,cmap='gray')
        figure2.canvas.header_visible = False 
        plt.show()

    output4 = widgets.Output()
    with output4:
        figure4, subplot4 = plt.subplots(figsize=(4,4))
        subplot4.imshow(initial_image,cmap='gray')
        figure4.canvas.header_visible = False 
        plt.show()


    def load_frames(dummy):
        global sinogram
        print("Loading sinogram from: ",sinogram_filepath)
        sinogram = np.load(sinogram_filepath ) 
        print(f'\t Loaded! Sinogram shape: {sinogram.shape}. Type: {type(sinogram)}' )
        selection_slider.widget.max, selection_slider.widget.value = sinogram.shape[0]-1, sinogram.shape[0]//2
        play_control.widget.max = selection_slider.widget.max
        widgets.interactive_output(update_imshow, {'sinogram':fixed(np.angle(sinogram)),'figure':fixed(figure),'subplot':fixed(subplot),'title':fixed('Obj Phase'),'cmap':fixed('hsv'), 'frame_number': selection_slider.widget})
        widgets.interactive_output(update_imshow, {'sinogram':fixed(np.abs(sinogram)),'figure':fixed(figure3),'subplot':fixed(subplot3),'title':fixed('Obj Mag'), 'frame_number': selection_slider.widget})

        probe =  np.load(probe_filepath) 
        widgets.interactive_output(update_imshow, {'sinogram':fixed(np.angle(probe)),'figure':fixed(figure2),'subplot':fixed(subplot2),'title':fixed('Probe Phase'), 'cmap':fixed('hsv'), 'frame_number': selection_slider.widget})
        widgets.interactive_output(update_imshow, {'sinogram':fixed(np.abs(probe)),'figure':fixed(figure4),'subplot':fixed(subplot4),'title':fixed('Probe Mag'), 'frame_number': selection_slider.widget})

    play_box, selection_slider,play_control = slide_and_play(label="Frame Selector")

    load_frames_button  = Button(description="Load Frames",layout=buttons_layout,icon='folder-open-o')
    load_frames_button.trigger(load_frames)

    buttons_box = widgets.Box([load_frames_button.widget],layout=get_box_layout('100%',align_items='center'))

    controls_box = widgets.Box([play_box],layout=get_box_layout('500px'))
    objects_box = widgets.HBox([output,output3,output2,output4])
    object_box = widgets.VBox([controls_box,objects_box])
    box = widgets.HBox([object_box])
    box = widgets.VBox([buttons_box,box])

    return box

def cropunwrap_tab():

    output = widgets.Output()
    with output:
        figure_unwrap, subplot_unwrap = plt.subplots(figsize=(5,5))
        subplot_unwrap.imshow(np.random.random((4,4)),cmap='gray')
        subplot_unwrap.set_title('Original')
        figure_unwrap.canvas.draw_idle()
        figure_unwrap.canvas.header_visible = False 
        plt.show()

    output2 = widgets.Output()
    with output2:
        figure_unwrap2, subplot_unwrap2 = plt.subplots(figsize=(5,5))
        subplot_unwrap2.imshow(np.random.random((4,4)),cmap='gray')
        subplot_unwrap2.set_title('Unwrapped')
        figure_unwrap2.canvas.draw_idle()
        figure_unwrap2.canvas.header_visible = False 
        plt.show()

    
    def load_frames(dummy):
        global sinogram
        
        print("Loading sinogram from: ",sinogram_filepath )
        sinogram = np.load(sinogram_filepath ) 
        print(f'\t Loaded! Sinogram shape: {sinogram.shape}. Type: {type(sinogram)}' )
        selection_slider.widget.max, selection_slider.widget.value = sinogram.shape[0]-1, sinogram.shape[0]//2
        play_control.widget.max = selection_slider.widget.max
        top_crop.widget.max  = bottom_crop.widget.max = sinogram.shape[1]//2 - 1
        left_crop.widget.max = right_crop.widget.max  = sinogram.shape[2]//2 - 1
        widgets.interactive_output(update_imshow, {'sinogram':fixed(np.angle(sinogram)),'figure':fixed(figure_unwrap),'subplot':fixed(subplot_unwrap),'title':fixed(''),'top': top_crop.widget, 'bottom': bottom_crop.widget, 'left': left_crop.widget, 'right': right_crop.widget, 'frame_number': selection_slider.widget})

    def preview_unwrap(dummy):
        cropped_frame = sinogram[selection_slider.widget.value,top_crop.widget.value:-bottom_crop.widget.value,left_crop.widget.value:-right_crop.widget.value]
        cropped_frame = cropped_frame[np.newaxis]
        unwrapped_frame = unwrap_phase(np.angle(cropped_frame))
        widgets.interactive_output(update_imshow, {'sinogram':fixed(unwrapped_frame),'figure':fixed(figure_unwrap2),'subplot':fixed(subplot_unwrap2),'title':fixed(''),'frame_number': fixed(0)})

    play_box, selection_slider,play_control = slide_and_play(label="Frame Selector")
    
    load_frames_button  = Button(description="Load Frames",layout=buttons_layout,icon='folder-open-o')
    load_frames_button.trigger(load_frames)

    preview_unwrap_button = Button(description="Preview Unwrap",layout=buttons_layout,icon='play') 
    preview_unwrap_button.trigger(preview_unwrap)
    
    buttons_box = widgets.Box([load_frames_button.widget,preview_unwrap_button.widget],layout=get_box_layout('100%',align_items='center'))
    sliders_box = widgets.Box([top_crop.widget,bottom_crop.widget,left_crop.widget,right_crop.widget],layout=sliders_box_layout)

    controls_box = widgets.Box([buttons_box,play_box,sliders_box],layout=get_box_layout('500px'))
    box = widgets.HBox([controls_box,vbar,output,output2])
    return box

############################################ DEPLOYMENT ######################################################

def deploy_tabs(tab1=inputs_tab(),tab2=center_tab(),tab3=reconstruction_tab(),tab4=cropunwrap_tab()):
    
    children_dict = {
    "Inputs"   : tab1,
    "Center"   : tab2,
    "Preview"  : tab3,
    "Unwrap"   : tab4}
    
    tab = widgets.Tab()
    tab.children = list(children_dict.values())
    for i in range(len(children_dict)): tab.set_title(i,list(children_dict.keys())[i]) # insert title in the tabs

    return tab, global_dict  



