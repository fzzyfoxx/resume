import ipywidgets as widgets
from IPython.display import display, clear_output
from IPython.core.display import HTML
import pandas as pd

class OutputSelectionWidgets:
    """
    A class for displaying and navigating through a list of options using interactive widgets.

    This class provides a user interface for selecting options from a list, with support for
    navigating between options, accepting a selection, and applying a callback function.

    Attributes:
        options: A list of options to display.
        current_index: The index of the currently displayed option.
        selected_option: The option selected by the user.
        call_func: A callback function to execute when an option is accepted.
        dark_mode_css: CSS styles for enabling dark mode in the widget display.
    """

    def __init__(self, call_func):
        """
        Initialize the OutputSelectionWidgets instance.

        Args:
            call_func: A callback function to execute when an option is accepted.
        """
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
        """
        Display the widget interface for navigating and selecting options.

        Args:
            options: A list of options to display.
        """
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
        """
        Update the widget display to show the current option.
        """
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
                display(widgets.Label(f"Option {self.current_index + 1} of {len(self.options)}"))

    def on_prev(self, _):
        """
        Navigate to the previous option in the list.
        """
        if self.current_index > 0:
            self.current_index -= 1
            self.update_display()

    def on_next(self, _):
        """
        Navigate to the next option in the list.
        """
        if self.current_index < len(self.options) - 1:
            self.current_index += 1
            self.update_display()

    def on_accept(self, _):
        """
        Accept the currently displayed option and execute the callback function.
        """
        self.selected_option = self.options[self.current_index]
        clear_output()
        print("Option accepted.")
        if self.call_func is not None:
            self.call_func()

    def get_selected_option(self):
        """
        Get the option selected by the user.

        Returns:
            The selected option.
        """
        return self.selected_option


class DisplayIter:
    """
    A class for iterating through a sequence of inputs and displaying them using OutputSelectionWidgets.

    This class allows users to interactively select options from a sequence and execute a target function
    when the iteration is complete.

    Attributes:
        display_widget: An instance of OutputSelectionWidgets for displaying options.
        options: A list of selected options.
        target_func: A function to execute when the iteration is complete.
        iterable: An iterator over the input sequence.
    """

    def __init__(self, inputs, target_func):
        """
        Initialize the DisplayIter instance.

        Args:
            inputs: A sequence of inputs to iterate through.
            target_func: A function to execute when the iteration is complete.
        """
        self.display_widget = OutputSelectionWidgets(call_func=self.__next__)
        self.options = []
        self.target_func = target_func

        self._set_iterable(inputs)

        next(self)

    def _set_iterable(self, inputs):
        """
        Set the input sequence as an iterable.

        Args:
            inputs: A sequence of inputs to iterate through.
        """
        self.iterable = iter(inputs)

    def __next__(self):
        """
        Display the next option in the sequence or execute the target function if the sequence is complete.
        """
        selected_option = self.display_widget.get_selected_option()
        if selected_option is not None:
            self.options.append(selected_option)
        try:
            next_option = next(self.iterable)
            self.display_widget.display(next_option)
        except StopIteration:
            self.target_func()