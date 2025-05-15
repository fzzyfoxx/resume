import ipywidgets as widgets
from IPython.display import display

class ChatInterface:
    """
    A class to create a chat interface for interacting with a chatbot within jupyter notebooks.
    This class provides a simple and interactive way to send messages to a chatbot and receive responses.
    It works with the SimpleChatBot class from fcgb.chatbot.
    Interface provides also a button to send 'button' action to the chatbot with a custom message (how message is handled is up to the chatbot implementation).
    If for a given thread_id there is some message_history in chatbot.memory they are displayed in the chat box (without the 'hidden' messages).

    Args:
    - chatbot: An instance of the SimpleChatBot class.
    - accept_button_msg: A string representing the message to be sent when the accept button is clicked.
    - thread_id: A string representing the thread ID for the current conversation.
    - template_inputs: A dictionary containing variables to be passed to the chatbot prompts templates.
    - height: An integer representing the height of the chat box in pixels.
    """
    def __init__(self, chatbot, accept_button_msg=None, thread_id=None, template_inputs={}, height=600, dark_mode=True):
        
        self.chatbot = chatbot
        self.thread_id = thread_id if thread_id else 'test_thread'
        self.accept_button_msg = accept_button_msg if accept_button_msg else "Accept"
        self.template_inputs = template_inputs
        
        # Create widgets
        self.chat_output = widgets.HTML(value="", layout=widgets.Layout(height=f"{height}px", overflow="auto", border="1px solid black", padding="10px"))
        self.user_input = widgets.Text(placeholder="Type your message here...", continous_update=False, layout=widgets.Layout(width="70%"))
        self.send_button = widgets.Button(description="Send", button_style="success")
        self.accept_button = widgets.Button(description="Accept", button_style="success")
        
        # Set up event handlers
        self.send_button.on_click(self._handle_send)
        self.accept_button.on_click(self._handle_accept)
        #self.user_input.on_submit(self._handle_send)
        self.user_input.continuous_update = False
        self.user_input.observe(self._handle_send, 'value')
        # Layout
        self.chat_box = widgets.VBox([self.chat_output, widgets.HBox([self.user_input, self.send_button, self.accept_button])])

        # Apply custom styles
        style = self._dark_mode_style() if dark_mode else self._light_mode_style()
        
        # Set custom CSS classes for widgets
        self.chat_output.add_class("custom-chat-box")
        self.chat_box.add_class("custom-chat-box")
        self.user_input.add_class("custom-input")
        self.send_button.add_class("custom-button")
        self.accept_button.add_class("custom-button")
        
        self.style_widget = widgets.HTML(value=style)
        # Display the chat interface
        display(self.style_widget)
        display(self.chat_box)

        # Initialize the chatbot thread
        self._init_bot()
    
    def _init_bot(self):
        """
        Initialize the chatbot thread
        If it already exists, this command will do nothing (chatbot take care of it internally)
        """
        self.chatbot.init_thread(self.thread_id, template_inputs=self.template_inputs)
        
        # Display existing messages
        messages = self.chatbot.get_messages(self.thread_id)
        if messages:
            for msg in messages:
                if msg[2] != 'hidden':
                    self._append_message(msg[0].upper(), msg[1])

    def _call_llm(self, human_message):
        """
        Invokes chatbot's graph with the human message passed in input box
        """
        response = self.chatbot.response(human_message, thread_id=self.thread_id)
        
        # Display chatbot response
        for sender, msg in response:
            self._append_message(sender.upper(), msg)
        
        # Clear input box
        self.user_input.value = ""

    def _handle_accept(self, _):
        """
        Actions for the accept button including:
        - Sending the accept button message to the chatbot
        - Closing the chat interface
        - Retreiving the final output of the chat process as summary variable
        """
        human_message = {"type": "button", "value": self.accept_button_msg}
        self._call_llm(human_message)

        self.style_widget.close()
        self.chat_box.close()

        self.summary = self.chatbot.get_summary(self.thread_id)
    
    def _handle_send(self, _):
        """
        Handler for the send button click event.
        """
        user_message = self.user_input.value.strip()
        if not user_message:
            return
        
        # Display user message
        self._append_message("HUMAN", user_message)
        
        # Process response from chatbot
        human_message = {"type": "message", "value": user_message}
        self._call_llm(human_message)
    
    def _append_message(self, sender, message):
        """
        Append a message to the chat output area.
        """
        formatted_message = "<br>" + "&nbsp;"*4 + message.replace("\n", "<br>" + "&nbsp;"*4)
        new_message = f"<div><b>{sender}:</b> {formatted_message}</div><hr style='border: 1px dashed #ccc;'>"
        self.chat_output.value += new_message

    def _dark_mode_style(self):
        return """
        <style>
            .cell-output-ipywidget-background {
            background-color: transparent !important;
            }
            .custom-chat-box {
            background-color: transparent !important;
            color: white !important;
            border: white !important;
            overflow: scroll;
            }
            .custom-input input {
            background-color: transparent !important;
            color: white !important;
            border: white !important;
            }
            .custom-button {
            background-color: green !important;
            color: white !important;
            }
        </style>
        """
    
    def _light_mode_style(self):
        return """
        <style>
            .cell-output-ipywidget-background {
            background-color: white !important;
            }
            .custom-chat-box {
            background-color: white !important;
            color: black !important;
            border: black !important;
            overflow: scroll;
            }
            .custom-input input {
            background-color: white !important;
            color: black !important;
            border: black !important;
            }
            .custom-button {
            background-color: green !important;
            color: white !important;
            }
        </style>
        """