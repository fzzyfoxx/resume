import ipywidgets as widgets
from IPython.display import display, clear_output
from IPython.core.display import HTML


dark_mode_css = """
<style>
.cell-output-ipywidget-background {
   background-color: transparent !important;
}
.custom-int-text input {
    background-color: lightgray !important;
    color: black !important;
    border: none !important;
}

.custom-int-text label {
    background-color: transparent !important;
    color: white !important;
    border: none !important;
}

.custom-checkbox input[type="checkbox"] {
    accent-color: lightgreen !important;
}

.widget-button {
    background-color: #444444;
    color: #ffffff;
    border: 1px solid #555555;
}
</style>
"""

def set_widget_title(widget, title):
    widget.layout.display = 'flex'
    widget.layout.flex_direction = 'column'
    widget.layout.align_items = 'center'
    widget.layout.justify_content = 'center'
    widget.children = [widgets.Label(value=title), widget.children[0]]
def create_widgets_for_dict(d, css=""):
    widgets_dict = {}
    for key, value in d.items():
        layout = widgets.Layout(width='300px', height='30px')
        style = {'description_width': 'initial'}

        if isinstance(value, dict):
            display_dict(value, css=css, nested=True, nested_title=key)
        else:
            if isinstance(value, bool):
                widget = widgets.Checkbox(value=value, description=key, layout=layout, style=style, _dom_classes=('custom-int-text','custom-checkbox'))
            elif isinstance(value, int):
                widget = widgets.IntText(value=value, description=key, layout=layout, style=style, _dom_classes=('custom-int-text',))
            elif isinstance(value, float):
                widget = widgets.FloatText(value=value, description=key, layout=layout, style=style, _dom_classes=('custom-int-text',))
            elif isinstance(value, str):
                widget = widgets.Text(value=value, description=key, layout=layout, style=style, _dom_classes=('custom-int-text',))
            else:
                continue

            widgets_dict[key] = widget
    return widgets_dict

def update_dict_from_widgets(d, widgets_dict):
    for key, widget in widgets_dict.items():
        d[key] = widget.value

def display_dict(d, trainer=None, compile_args_func=None, compile_args={}, css=dark_mode_css, nested=False, nested_title=None):

    if nested:
        css = f'<h5> {nested_title} </h5>'
    display(HTML(css))

    widgets_dict = create_widgets_for_dict(d, css=css)
    widget_list = list(widgets_dict.values())
    
    # Create a GridBox to display widgets in 4 columns
    grid = widgets.GridBox(widget_list, layout=widgets.Layout(
        grid_template_columns="repeat(4, 1fr)",
        grid_gap="10px 10px"
    ))
    
    display(grid)
    
    update_button = widgets.Button(description="Accept Parameters" if not nested else "Pass Parameters")
    display(update_button)
    
    def on_update_button_clicked(b):
        update_dict_from_widgets(d, widgets_dict)
        print("Dictionary updated:", d)

    def trainer_compile(b):
        trainer.compile_model(
                    model_args = d, 
                    print_summary = True,
                    summary_kwargs = {'expand_nested': False, 'line_length': 100},
                    **compile_args_func(**compile_args)
                )
        print('Model compiled. Use trainer.train_model for training.\nAccept parameters again to re-compile.')
    
    update_button.on_click(on_update_button_clicked)

    if trainer is not None:
        update_button.on_click(trainer_compile)