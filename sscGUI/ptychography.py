import os, json, ast, h5py
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import getpass

import ipywidgets as widgets 
from ipywidgets import fixed

from .misc import Button, Input, update_imshow, slide_and_play

# import sscCdi, sscPimega, sscRaft, sscRadon, sscResolution
# from sscCdi import create_propagation_video
# from .cat_ptycho_processing import masks_application
# from ..misc import miqueles_colormap
# from ..processing.unwrap import phase_unwrap
# from ..misc import create_directory_if_doesnt_exist

############################################  PATH DEFINITIONS ###########################################################################

username = getpass.getuser()

inputs_folder = "./example/inputs/"
output_folder = "./example/outputs/"

template_dict_path = os.path.join(inputs_folder,inputs_folder)
output_dict_path = os.path.join(output_folder, f'{username}_input_dict.json') 

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

    save_on_click_partial = partial(save_on_click,output_dict_filepath=output_folder,dictionary=global_dict)
    saveJsonButton = Button(description="Save Inputs",layout=buttons_layout,icon='fa-floppy-o')
    saveJsonButton.trigger(save_on_click_partial)

    """ Create widgets """

    label1 = create_label_widget("Data Selection")
    data_folder_str     = Input(global_dict,"data_folder",description="Proposal Path",layout=items_layout2)
    
    label2 = create_label_widget("Diffraction Pattern")
    global center_y, center_x
    center_x    = Input({'dummy-key':global_dict["DP_center"][1]},'dummy-key',bounded=(0,3072,1),slider=True,description="Center column (x)",layout=slider_layout)
    center_y    = Input({'dummy-key':global_dict["DP_center"][0]},'dummy-key',bounded=(0,3072,1),slider=True,description="Center row (y)   ",layout=slider_layout)
    
    label6 = create_label_widget("Post-processing")
    phase_unwrap      = Input({'dummy-key':global_dict["phase_unwrap"]},'dummy-key',description="Phase Unwrap",layout=checkbox_layout)
    phase_unwrap_box = widgets.Box([phase_unwrap.widget],layout=items_layout2)
    global top_crop, bottom_crop,left_crop,right_crop # variables are reused in crop tab
    top_crop      = Input({'dummy_key':0},'dummy_key',bounded=(0,10,1), description="Top", slider=True,layout=slider_layout)
    bottom_crop   = Input({'dummy_key':1},'dummy_key',bounded=(1,10,1), description="Bottom", slider=True,layout=slider_layout)
    left_crop     = Input({'dummy_key':0},'dummy_key',bounded=(0,10,1), description="Left", slider=True,layout=slider_layout)
    right_crop    = Input({'dummy_key':1},'dummy_key',bounded=(1,10,1), description="Right", slider=True,layout=slider_layout)

    def update_global_dict(data_folder_str,center_y,center_x,phase_unwrap):
        global global_dict
        global_dict["data_folder"]  = data_folder_str
        global_dict["DP_center"]    = [center_y,center_x]
        global_dict["phase_unwrap"] = phase_unwrap

    """ Monitor variable and call function when they change """
    widgets.interactive_output(update_global_dict,{'data_folder_str':data_folder_str.widget,
                                                    # energy
                                                    # distance
                                                    'center_y':center_y.widget,
                                                    'center_x':center_x.widget,
                                                    'phase_unwrap': phase_unwrap.widget,
                                                    'top_crop': top_crop.widget,
                                                    'bottom_crop': bottom_crop.widget,
                                                    'left_crop': left_crop.widget,
                                                    'right_crop': right_crop.widget,
                                                     })

    box = widgets.Box([label1,data_folder_str.widget,label2,center_x.widget,center_y.widget,label6,phase_unwrap_box],layout=box_layout)

    return box

def center_tab():

    # output = widgets.Output()
    # with output:
    #     figure, subplot = plt.subplots(figsize=(5,5),constrained_layout=True)
    #     figure,subplot.imshow(np.random.random((4,4)))
    #     subplot.set_title('Diffraction Pattern')
    #     figure.canvas.header_visible = False 
    #     plt.show()


    # def plotshow(figure,subplot,image,title="",figsize=(8,8),savepath=None,show=False):
    #     subplot.clear()
    #     cmap, colors, bounds, norm = miqueles_colormap(image)
    #     handle = subplot.imshow(image, interpolation='nearest', cmap = cmap, norm=norm)
    #     if title != "":
    #         subplot.set_title(title)
    #     if show:
    #         if __name__ == '__main__': 
    #             plt.show()

    #     figure.canvas.draw_idle()

    # def update_mask(figure, subplot,output_dictionary,image,key1,key2,key3,cy,cx,button,exposure,exposure_time,radius):

    #     output_dictionary[key1] = [cy,cx]
    #     output_dictionary[key2] = [button,radius]
    #     output_dictionary[key3] = [exposure,exposure_time]
    #     plotshow(figure,subplot,image)

    # def load_difpad(dummy):

    #     mdata_filepath = os.path.join(global_dict["data_folder"],global_dict['acquisition_folders'][0],'mdata.json')
    #     input_dict = json.load(open(mdata_filepath))

    #     image = np.load(global_paths_dict['flipped_difpad_filepath'])
    #     widgets.interactive_output(update_mask,{'figure':fixed(figure), 'subplot': fixed(subplot),
    #                                             'output_dictionary':fixed(global_dict),'image':fixed(image),
    #                                             'key1':fixed('DP_center'),'key2':fixed('central_mask'),
    #                                             'cy':center_y.widget,'cx':center_x.widget,
    #                                             'button':central_mask_bool.widget,
    #                                             'exposure_time':fixed(input_dict['/entry/beamline/detector']['pimega']["exposure time"]),
    #                                             'radius':central_mask_radius.widget})

    # load_difpad_button  = Button(description="Load Diffraction Pattern",layout=buttons_layout,icon='folder-open-o')
    # load_difpad_button.trigger(load_difpad)

    # """ Difpad center boxes """
    # sliders_box = widgets.HBox([center_y.widget,center_x.widget,central_mask_radius.widget],layout=box_layout)
    # controls = widgets.Box([load_difpad_button.widget,sliders_box,central_mask_bool.widget],layout=get_box_layout('500px'))
    # box = widgets.HBox([controls,vbar,output])
    # return box
    pass

def fresnel_tab():
    
    # image_list, fresnel_number_list = [np.random.random((5,5))], [0]

    # output = widgets.Output()
    # with output:
    #     figure, subplot = plt.subplots(figsize=(4,4))
    #     subplot.imshow(image_list[0],cmap='jet') # initialize
    #     subplot.set_title('Propagated Probe')
    #     figure.canvas.header_visible = False 
    #     plt.show()


    # def update_probe_plot(fig,subplot,image_list,frame_list,index):
    #     subplot.clear()
    #     subplot.set_title(f'Frame #: {frame_list[index]:.1e}')
    #     subplot.imshow(image_list[index],cmap='jet')
    #     fig.canvas.draw_idle()

    # def on_click_propagate(dummy):
    
    #     print('Propagating probe...')
    #     image_list, fresnel_number_list = create_propagation_video(global_paths_dict['probe_filepath'],
    #                                                     starting_f_value=starting_f_value,
    #                                                     ending_f_value=ending_f_value,
    #                                                     number_of_frames=number_of_frames,
    #                                                     jupyter=True)
        
    #     play_control.widget.max, selection_slider.widget.max = len(image_list)-1, len(image_list)-1

    #     widgets.interactive_output(update_probe_plot,{'fig':fixed(figure),'subplot':fixed(subplot),'image_list':fixed(image_list),'frame_list':fixed(fresnel_number_list),'index':selection_slider.widget})
    #     print('\t Done!')

    # def update_values(n_frames,start_f,end_f,power):
    #     global starting_f_value, ending_f_value, number_of_frames
    #     starting_f_value=-start_f*10**power
    #     ending_f_value=-end_f*10**power
    #     number_of_frames=int(n_frames)
    #     label.value = r"Propagating from f = {0}$\times 10^{{{1}}}$ to {2}$\times 10^{{{1}}}$".format(start_f,power,end_f)

    # play_box, selection_slider,play_control = slide_and_play(label="")

    # power   = Input( {'dummy-key':-4}, 'dummy-key', bounded=(-10,10,1),  slider=True, description=r'Exponent'       ,layout=items_layout)
    # start_f = Input( {'dummy-key':-1}, 'dummy-key', bounded=(-10,0,1),   slider=True, description='Start f-value'   ,layout=items_layout)
    # end_f   = Input( {'dummy-key':-9}, 'dummy-key', bounded=(-10,0,1),   slider=True, description='End f-value'     ,layout=items_layout)
    # n_frames= Input( {'dummy-key':100},'dummy-key', bounded=(10,200,10), slider=True, description='Number of Frames',layout=items_layout)

    # label = widgets.Label(value=r"Propagating from f = {0} $\times 10^{{{1}}}$ to {2} $\times 10^{{{1}}}$".format(start_f,power,end_f),layout=items_layout)

    # widgets.interactive_output(update_values,{'n_frames':n_frames.widget,'start_f':start_f.widget,'end_f':end_f.widget,'power':power.widget})
    # propagate_button = Button(description=('Propagate Probe'),layout=buttons_layout)
    # propagate_button.trigger(on_click_propagate)

    # box = widgets.Box([n_frames.widget, power.widget, start_f.widget,end_f.widget,label,propagate_button.widget,fresnel_number.widget],layout=get_box_layout('700px'))
    # play_box = widgets.VBox([play_box,output],layout=box_layout)
    # box = widgets.HBox([box,vbar,play_box])
    # return box
    pass

def cropunwrap_tab():

    # output = widgets.Output()
    # with output:
    #     figure_unwrap, subplot_unwrap = plt.subplots(figsize=(3,3))
    #     subplot_unwrap.imshow(np.random.random((4,4)),cmap='gray')
    #     subplot_unwrap.set_title('Cropped image')
    #     figure_unwrap.canvas.draw_idle()
    #     figure_unwrap.canvas.header_visible = False 
    #     plt.show()

    # output2 = widgets.Output()
    # with output2:
    #     figure_unwrap2, subplot_unwrap2 = plt.subplots(figsize=(3,3))
    #     subplot_unwrap2.imshow(np.random.random((4,4)),cmap='gray')
    #     subplot_unwrap2.set_title('Unwrapped image')
    #     figure_unwrap2.canvas.draw_idle()
    #     figure_unwrap2.canvas.header_visible = False 
    #     plt.show()

    
    # def load_frames(dummy):
    #     global sinogram
        
    #     print("Loading sinogram from: ",global_paths_dict["sinogram_filepath"] )
    #     sinogram = np.load(global_paths_dict["sinogram_filepath"] ) 
    #     print(f'\t Loaded! Sinogram shape: {sinogram.shape}. Type: {type(sinogram)}' )
    #     selection_slider.widget.max, selection_slider.widget.value = sinogram.shape[0]-1, sinogram.shape[0]//2
    #     play_control.widget.max = selection_slider.widget.max
    #     top_crop.widget.max  = bottom_crop.widget.max = sinogram.shape[1]//2 - 1
    #     left_crop.widget.max = right_crop.widget.max  = sinogram.shape[2]//2 - 1
    #     widgets.interactive_output(update_imshow, {'sinogram':fixed(np.angle(sinogram)),'figure':fixed(figure_unwrap),'subplot':fixed(subplot_unwrap),'title':fixed(True),'top': top_crop.widget, 'bottom': bottom_crop.widget, 'left': left_crop.widget, 'right': right_crop.widget, 'frame_number': selection_slider.widget})


    # def preview_unwrap(dummy):
    #     cropped_frame = sinogram[selection_slider.widget.value,top_crop.widget.value:-bottom_crop.widget.value,left_crop.widget.value:-right_crop.widget.value]
    #     cropped_frame = cropped_frame[np.newaxis]
    #     unwrapped_frame = phase_unwrap(np.angle(cropped_frame),iterations=0,non_negativity=False,remove_gradient = False)
    #     widgets.interactive_output(update_imshow, {'sinogram':fixed(unwrapped_frame),'figure':fixed(figure_unwrap2),'subplot':fixed(subplot_unwrap2),'title':fixed(True),'frame_number': fixed(0)})

    # play_box, selection_slider,play_control = slide_and_play(label="Frame Selector")
    
    # load_frames_button  = Button(description="Load Frames",layout=buttons_layout,icon='folder-open-o')
    # load_frames_button.trigger(load_frames)

    # preview_unwrap_button = Button(description="Preview Unwrap",layout=buttons_layout,icon='play') 
    # preview_unwrap_button.trigger(preview_unwrap)
    
    # buttons_box = widgets.Box([load_frames_button.widget,preview_unwrap_button.widget],layout=get_box_layout('100%',align_items='center'))
    # sliders_box = widgets.Box([top_crop.widget,bottom_crop.widget,left_crop.widget,right_crop.widget],layout=sliders_box_layout)

    # controls_box = widgets.Box([buttons_box,play_box,sliders_box],layout=get_box_layout('500px'))
    # box = widgets.HBox([controls_box,vbar,output,output2])
    # return box
    pass

def reconstruction_tab():
    
    # initial_image = np.random.random((5,5)) # dummy

    # output = widgets.Output()
    # with output:
    #     figure, subplot = plt.subplots(figsize=(4,4))
    #     subplot.imshow(initial_image,cmap='gray')
    #     figure.canvas.header_visible = False 
    #     plt.show()


    # output3 = widgets.Output()
    # with output3:
    #     figure3, subplot3 = plt.subplots(figsize=(4,4))
    #     subplot3.imshow(initial_image,cmap='gray')
    #     figure3.canvas.header_visible = False 
    #     plt.show()


    # output2 = widgets.Output()
    # with output2:
    #     figure2, subplot2 = plt.subplots(figsize=(4,4))
    #     subplot2.imshow(initial_image,cmap='gray')
    #     figure2.canvas.header_visible = False 
    #     plt.show()


    # def load_frames(dummy):
    #     global sinogram
    #     print("Loading sinogram from: ",global_paths_dict["sinogram_filepath"] )
    #     sinogram = np.load(global_paths_dict["sinogram_filepath"] ) 
    #     print(f'\t Loaded! Sinogram shape: {sinogram.shape}. Type: {type(sinogram)}' )
    #     selection_slider.widget.max, selection_slider.widget.value = sinogram.shape[0]-1, sinogram.shape[0]//2
    #     play_control.widget.max = selection_slider.widget.max
    #     widgets.interactive_output(update_imshow, {'sinogram':fixed(np.angle(sinogram)),'figure':fixed(figure),'subplot':fixed(subplot),'title':fixed(True), 'frame_number': selection_slider.widget})
    #     widgets.interactive_output(update_imshow, {'sinogram':fixed(np.abs(sinogram)),'figure':fixed(figure3),'subplot':fixed(subplot3),'title':fixed(True), 'frame_number': selection_slider.widget})

    #     probe = np.abs(np.load(global_paths_dict["probe_filepath"]))[:,0,:,:] # get only 0th order 
    #     widgets.interactive_output(update_imshow, {'sinogram':fixed(probe),'figure':fixed(figure2),'subplot':fixed(subplot2),'title':fixed(True), 'cmap':fixed('jet'), 'frame_number': selection_slider.widget})

    # play_box, selection_slider,play_control = slide_and_play(label="Frame Selector")

    # load_frames_button  = Button(description="Load Frames",layout=buttons_layout,icon='folder-open-o')
    # load_frames_button.trigger(load_frames)

    # buttons_box = widgets.Box([load_frames_button.widget],layout=get_box_layout('100%',align_items='center'))

    # controls_box = widgets.Box([play_box],layout=get_box_layout('500px'))
    # objects_box = widgets.HBox([output,output3,output2])
    # object_box = widgets.VBox([controls_box,objects_box])
    # box = widgets.HBox([object_box])
    # box = widgets.VBox([buttons_box,box])

    # return box
    pass


############################################ DEPLOYMENT ######################################################

def deploy_tabs(tab1=inputs_tab(),tab2=center_tab(),tab3=fresnel_tab(),tab4=reconstruction_tab(),tab5=cropunwrap_tab()):
    
    children_dict = {
    "Inputs"            : tab1,
    "Find Center"       : tab2,
    "Probe Propagation" : tab3,
    "Crop and Unwrap"   : tab4,
    "Reconstruction"    : tab5
    }
    
    gpus = Input({'dummy_key':1}, 'dummy_key',bounded=(1,5,1),  slider=True,description="Insert # of gpus to use:")
    cpus = Input({'dummy_key':32},'dummy_key',bounded=(1,32,1),slider=True,description="Insert # of CPUs to use:")
    box = widgets.HBox([gpus.widget,cpus.widget])

    tab = widgets.Tab()
    tab.children = list(children_dict.values())
    for i in range(len(children_dict)): tab.set_title(i,list(children_dict.keys())[i]) # insert title in the tabs

    return box,tab, global_dict  



