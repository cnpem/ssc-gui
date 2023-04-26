import ast
import ipywidgets as widgets 
from ipywidgets import fixed 


field_style = {'description_width': 'initial'}

def update_imshow(sinogram,figure,subplot,frame_number,top=0, bottom=None,left=0,right=None,axis=0,title='',clear_axis=True,cmap='gray',norm=None):
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
    if title != '':
        subplot.set_title(title)
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

    def __init__(self,type,dictionary,key,description="",layout=None,bounded=()):
        
        self.dictionary = dictionary
        self.key = key
        
        if layout == None: 
            self.items_layout = widgets.Layout()
        else:
            self.items_layout = layout
   
        if type == 'Checkbox':
            self.widget = widgets.Checkbox(description=description,value=self.dictionary[self.key],layout=self.items_layout, style=field_style)
        elif type == "IntSlider":
            self.widget = widgets.IntSlider(description=description,value=self.dictionary[self.key],layout=self.items_layout, style=field_style,min=bounded[0],max=bounded[1],step=bounded[2])
        elif type == "IntText":
            self.widget = widgets.IntText(description=description,value=self.dictionary[self.key],layout=self.items_layout, style=field_style)
        elif type == "BoundedIntText":
            self.widget = widgets.BoundedIntText(description=description,value=self.dictionary[self.key],layout=self.items_layout, style=field_style, min=bounded[0],max=bounded[1],step=bounded[2])
        elif type == "FloatText":
            self.widget = widgets.FloatText(description=description,value=self.dictionary[self.key],layout=self.items_layout, style=field_style)
        elif type == "BoundedFloatText":
            self.widget = widgets.BoundedFloatText(description=description,value=self.dictionary[self.key],layout=self.items_layout, style=field_style,min=bounded[0],max=bounded[1],step=bounded[2])
        elif type == "TextString":
            self.widget = widgets.Text(description=description,value=self.dictionary[self.key],layout=self.items_layout, style=field_style)
        elif type == "TextList":
            self.widget = widgets.Text(description=description,value=str(self.dictionary[self.key]),layout=self.items_layout, style=field_style)
        elif type == "TextDict":
            self.widget = widgets.Text(description=description,value=str(self.dictionary[self.key]),layout=self.items_layout, style=field_style)
        
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

    selection_slider = Input('IntSlider',{"dummy_key":1},"dummy_key",description=description, bounded=(0,100,1),layout=widgets.Layout(width='max-width'))
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

