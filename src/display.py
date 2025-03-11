import ipywidgets as widgets
from IPython.display import display, clear_output
from IPython.core.display import HTML
import pandas as pd

class OutputSelectionWidgets:
    def __init__(self, call_func):
        self.options = []
        self.current_index = 0
        self.selected_option = None

        self.call_func = call_func

        self.dark_mode_css = """
        <style>
        .cell-output-ipywidget-background {
        background-color: transparent !important;
        }
        .overlay-text {
            color: white !important;
        }
        .custom-int-text input {
            background-color: lightgray !important;
            color: white !important;
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

        .widget-label {
            color: #e0e0e0 !important;
        }

        .widget-button {
            background-color: #444444;
            color: #ffffff;
            border: 1px solid #555555;
        }
        </style>
        """

    def display(self, options):
        self.options = options
        self.current_index = 0
        self.selected_option = None

        self.prev_button = widgets.Button(description="Previous")
        self.next_button = widgets.Button(description="Next")
        self.accept_button = widgets.Button(description="Accept")
        self.output = widgets.Output()

        self.prev_button.on_click(self.on_prev)
        self.next_button.on_click(self.on_next)
        self.accept_button.on_click(self.on_accept)

        self.update_display()

        display(HTML(self.dark_mode_css))
        display(widgets.HBox([self.prev_button, self.next_button, self.accept_button]))
        display(self.output)

    def update_display(self):
        with self.output:
            clear_output()
            if self.options:
                option = self.options[self.current_index]
                df = pd.DataFrame(option)
                
                df = df.style.map(lambda x: 'color: white').set_table_styles( # type: ignore
                        [
                            {'selector': 'td', 'props': [('white-space', 'pre-wrap'), ('word-wrap', 'break-word')]},
                            {'selector': 'th', 'props': [('color', 'white')]}
                        ]
                    )
                display(df)
                #print(f"\033[97mOption {self.current_index + 1} of {len(self.options)}\033[0m")
                display(widgets.Label(f"Option {self.current_index + 1} of {len(self.options)}"))

    def on_prev(self, _):
        if self.current_index > 0:
            self.current_index -= 1
            self.update_display()

    def on_next(self, _):
        if self.current_index < len(self.options) - 1:
            self.current_index += 1
            self.update_display()

    def on_accept(self, _):
        self.selected_option = self.options[self.current_index]
        clear_output()
        print("Option accepted.")
        if self.call_func is not None:
            self.call_func()

    def get_selected_option(self):
        return self.selected_option



class DisplayIter:
    def __init__(self, inputs, target_func):

        self.display_widget = OutputSelectionWidgets(call_func=self.__next__)
        self.options = []
        self.target_func = target_func

        self._set_iterable(inputs)

        next(self)

    def _set_iterable(self, inputs):
        self.iterable = iter(inputs)

    def __next__(self):
        selected_option = self.display_widget.get_selected_option()
        if selected_option is not None:
            self.options.append(selected_option)
        try:
            next_option = next(self.iterable)
            self.display_widget.display(next_option)
        except:
            self.target_func()