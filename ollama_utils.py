### ollama_utils
## Defines functions needed to setup and query an LM with an Ollama client.
## Based on README instructions of Ollama Python library | https://github.com/ollama/ollama-python

import re
from re import Match
import ollama
from ollama import (
    Client, 
    ChatResponse, 
    ListResponse,
    ProgressResponse
)
from typing import (
    List,
    Pattern
)

from logger import (
    logger,
    with_spinner
)

## Define constants
# LM | Must be available in Ollama library (https://ollama.com/library)
lm_name: str = 'qwen3:0.6b'

# URL | Ollama server url from Docker setup
url: str = 'http://localhost:11434'

# Message | Example message to send to LM
message: str = 'Why is the sky blue?'


class OllamaClient:
    """
    An Ollama client that can be used to chat with LMs.

    The user can chat with an LM by initializing this class then using the get_response method for a given LM and message.

    For example, to initialize the client for a given url:
    ```python
    url = 'my_url'
    client = OllamaClient(url=url)
    ```

    Then to get a response for a given message from a given LM:
    ```python
    lm_name = 'my_model'
    message = 'My message'

    client.get_response(lm_name=lm_name, message=message)
    ```
    
    Attributes
    ------------
        url: str, Optional
            The URL on which to host the Ollama client
            Defaults to 'http://localhost:11434'
        client: Client (Ollama Python)
            The Ollama client to use to get responses
    """
    def __init__(
        self, 
        url: str = url
    ) -> None:
        """
        Initialize the Ollama client hosted on the given url.
        
        Args
        ------------
            url: str, Optional
                The url on which to host the Ollama client
                Defaults to 'http://localhost:11434'
            
        Raises
        ------------
            Exception: 
                If initialization fails, error is logged and raised
        """
        try:
            self.url = url
            self.client = self._init_client()
        except Exception as e:
            logger.info(f'❌ Problem initializing Ollama client: {str(e)}')
            raise


    ## Connect to the Ollama client
    def _init_client(
        self
    ) -> Client:
        """
        Connect the Ollama client.
        
        Returns
        ------------
            Client (Ollama Python): 
                The Ollama client instance
            
        Raises
        ------------
            Exception: 
                If client connection fails, error is logged and raised
        """
        logger.info(f'⚙️ Starting Ollama client on URL `{self.url}`')
        try:
            # Create the client
            client = Client(host=self.url)
            
            # Test the connection with list and validate response
            response = client.list()
            
            # Check if `models` attribute exists
            if not hasattr(response, 'models'):
                raise ValueError("The response from client.list() is missing the 'models' attribute")
            
            # Check that `models` attribute is a list
            if not isinstance(response.models, list):
                raise ValueError("The 'models' attribute of client.list() should be a list")
            
            logger.info(f"✅ Ollama client connected successfully. Found {len(response.models)} models available.")
            return client
        except Exception as e:
            logger.error(f"❌ Failed connecting to Ollama client at {self.url}: {str(e)}")
            raise


    ## Check existing Ollama models
    def _list_pulled_models(
        self
    ) -> List[str | None]:
        """
        List all models available in Ollama storage.

        Returns
        ------------
            List[str | None]: 
                A list of all the available models
            
        Raises
        ------------
            Exception: 
                If getting list fails, error is logged and raised
        """
        try:
            # List all models available with Ollama
            ollama_models: ListResponse = ollama.list()
            # List all model names
            model_names: List[str | None] = [
                model.model for model in ollama_models.models
            ]
            logger.info(f'📝 Existing models `{model_names}`')
            return model_names
        except Exception as e:
            logger.error(f'❌ Problem listing models available in Ollama: `{str(e)}`')
            raise


    ## Pull LM
    def _pull_lm(
        self,
        lm_name: str = lm_name
    ) -> ProgressResponse:
        """
        Pull LM with given name from Ollama library if not already pulled (https://ollama.com/library).

        Args
        ------------
            lm_name: str, Optional 
                Name of the LM to pull from Ollama
                Defaults to 'qwen3:0.6b'
                Must be available in Ollama library (https://ollama.com/library)
            
        Raises
        ------------
            Exception: 
                If LM pull fails, error is logged and raised
        """
        response = ollama.pull(lm_name)
        return response


    ## Setup LM
    def _init_lm(
        self, 
        lm_name: str = lm_name
    ) -> None:
        """
        Initialize LM by checking that it's avaible from Ollama data.
        Pulls the model is it isn't avaialable.

        Args
        ------------
            lm_name: str, Optional 
                Name of the LM to initialize from Ollama
                Defaults to 'qwen3:0.6b'
                Must be available in Ollama library (https://ollama.com/library)
            
        Raises
        ------------
            Exception: 
                If LM initialization fails, error is logged and raised
        """
        logger.info(f'⚙️ Setting up Ollama model for LM `{lm_name}`')
        try:
            # Check existing models in Ollama
            model_names = self._list_pulled_models()

            # If `lm_name` not in model_names, pull it from Ollama
            if lm_name not in model_names:
                logger.info(f'⚙️ Pulling `{lm_name}` from Ollama')

                # Show spinner during the pulling process
                with with_spinner(description=f"⚙️ Pulling LM..."):
                    self._pull_lm(lm_name=lm_name)

                logger.info(f'✅ Model `{lm_name}` pulled from Ollama')
        except Exception as e:
            logger.error(f'❌ Problem getting ollama model: `{str(e)}`')
            raise


    ## Clean LM response
    def _remove_think_tags(
        self, 
        text: str | None
    ) -> str:
        """
        Remove <think></think> tags and all text within the tags.

        Args
        ------------
            text: str 
                Text to remove <think></think> tags from

        Returns
        ------------
            str: 
                The resulting string without the <think></think> tags and all the text within
            
        Raises
        ------------
            Exception: 
                If cleaning the text fails, error is logged and raised
        """
        try:
            # Find <think>...</think> within the text
            think_tag_pattern: Pattern[str] = re.compile(r'<think>.*?</think>\s*', re.DOTALL)
            # Make sure text is str
            # text can be typed as Optional[str] because of response.message.content type in Ollama Python library
            if text is None:
                error_message: str = '❌ Problem cleaning response, check error logs for details.'
                logger.error(error_message)
                raise ValueError(error_message)
            else:
                # If the tag is not fully closed, we handle that separately
                if not think_tag_pattern:
                    # Find any instances of <think> and </think> and substitute them with empty strings
                    outside_tags: str = re.sub(r'</?think>', '', text).strip()

                # If matched and closed, clean up tags from outside
                else:
                    # Change the <think>...</think> to an empty string
                    cleaned_text: str = think_tag_pattern.sub('', text).strip()
                    # Make sure no other tags remain
                    outside_tags = re.sub(r'</?think>', '', cleaned_text).strip()

                return outside_tags
        except Exception as e:
            logger.error(f'❌ Failed to clean text: `{str(e)}`')
            raise


    ## Query the LM
    def get_response(
        self, 
        lm_name: str = lm_name, 
        message: str = message,
        options: dict | None = None
    ) -> str | None:
        """
        Get a response for the given message from the LM with the given lm_name.

        For example, to get a response first initalize the class then use the get_response method:
        ```python
        url = 'my_url'
        lm_name = 'my_model'
        message = 'My Message'

        client = OllamaClient(url=url)
        client.get_response(lm_name=lm_name, message=message)
        ```

        Args
        ------------
            lm_name: str, Optional 
                Name of the LM to pull from Ollama
                Defaults to 'qwen3:0.6b'
                Must be available in Ollama library (https://ollama.com/library)
            message: str, Optional
                Message to send to LM and get response
                Defaults to 'Why is the sky blue?'
            options: dict, Optional
                Extra options to pass to the Ollama Client chat method
                Defaults to None

        Returns
        ------------
            str: 
                The LM response (without any thinking context)
                To keep thinking context, refrain from using the `self._remove_think_tags` method
            
        Raises
        ------------
            Exception: 
                If getting a response fails, error is logged and raised
        """
        logger.info(f'⚙️ Getting LM response for message `{message}`')
        ## Make sure LM is available
        try:
            self._init_lm(lm_name=lm_name)
        except Exception as e:
            logger.error(f'❌ Problem pulling LM `{lm_name}` from Ollama: `{str(e)}`')
            raise

        ## Get LM response
        try:
            with with_spinner(description=f"🔮 LM working its magic..."):
                # Format the message properly
                messages: dict = {
                    'role': 'user',
                    'content': message,
                }
                # Send a message to the model
                response: ChatResponse = self.client.chat(
                    model=lm_name, 
                    messages=[messages],
                    options=options
                )

            ## If no response, raise error
            error_message = '❌ Problem returning response, check error logs for details.'
            if response is None:
                logger.error(error_message)
                raise ValueError(error_message)

            ## Else return the response
            else:
                #content = response.message.content
                # If you want to keep LM's thinking context, comment out line below and use line above instead
                content = self._remove_think_tags(response.message.content)
                logger.info(f'📝 LM response `{content}`\n')
                return content
        except Exception as e:
            logger.error(f'❌ Problem getting LM response: `{str(e)}`')
            raise