### tests/test_unit.py
## Defines unit tests for methods in ./ollama_utils.py

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

## I'm learning how to create testing suites with this file, so lots of 
## (probably obvious for a seasoned developer) comments are included 
## to clarify my confusion

## For unit testing, we don't want to rely on real Ollama API calls, we just want 
## to test the logic of our code
# Patch is used to define where mocks (fake functions, classes, clients, etc.) 
# should replace the real thing in `ollama_utils.py`
# MagicMock is used to define the mocks that will replace the real things
# Bottom most patch goes with inner most argument, then next patch goes to next argument, and so forth
import unittest
from unittest.mock import patch, MagicMock
from ollama_utils import OllamaClient, lm_name, message

## Define constants
model_name = 'model-name'

## Mock response objects from Ollama Python library
# Mock Ollama Model
class MockModel:
    """
    A mock model object representing a pulled model from Ollama.
    This mocks an Ollama `Model` object which has a `model` attribute.
    
    This class simulates the structure of a model returned by ollama.list(),
    containing basic model information for testing purposes.
    
    Attributes
    ------------
        model: str
            The name of the model
    """
    def __init__(self, model):
        self.model = model


# Mock Ollama ListResponse
class MockListResponse:
    """
    A mock response object representing the result of an ollama.list() call.
    This mocks an Ollama `ListResponse` object which has a `models` attribute.
    This `models` attribute is a list of Ollama Model objects which will be mocked by 
    the `MockModel` class.
    
    This class simulates the structure of a list response from Ollama's list API,
    containing a collection of models for testing purposes.
    
    Attributes
    ------------
        models: List[MockModel]
            List of MockModel objects representing pulled models
    """
    def __init__(self, models):
        self.models = [MockModel(model=m) for m in models]


# Mock Ollama ChatResponse
class MockChatResponse:
    """
    A mock response object representing the result of an Ollama Client.chat() call.
    This mocks an Ollama `ChatResponse` object which has a `message` attribute. 
    The response of the LM is given by the `content` attribute of `message`.
    
    This class simulates the structure of a chat response from Ollama's chat API,
    containing message content for testing purposes.
    
    Attributes:
        message: MagicMock
            Mock object representing the chat message
    """
    def __init__(self, content):
        self.message = MagicMock()
        self.message.content = content


# Mock Ollama pull response (ProgressResponse)
class MockPullResponse:
    """
    A mock response object representing the result of an ollama.pull() call.
    This mocks an Ollama `ProgressResponse` object which has a `status` attribute.
    
    This class simulates the structure of a pull response from Ollama's pull API,
    containing status content for testing purposes.
    
    Attributes:
        status: str
            The status of the Ollama pull
    """
    def __init__(self, status):
        self.status = status


## Now let's test everything
class TestOllamaClientUnit(unittest.TestCase):
    """
    Unit tests for `OllamaClient` class.
    
    This test suite contains unit tests for the `OllamaClient` class of 
    `ollama_utils.py`, covering initialization, model management, response handling, 
    and utility methods.

    All tests use mocking to isolate the class under test from external dependencies.
    """


    ## Test successful client initialization
    # Utilize patch to replace all Ollama Client calls in `ollama_utils.py` with mock
    @patch('ollama_utils.Client')
    def test_init_client_success(self, mock_client):
        """
        Test successful initialization of OllamaClient with a custom URL.
        
        Verifications
        ------------
            The client is initialized with the correct URL
            The underlying Ollama Client is instantiated with the provided host
            The correct host parameter is passed to the Client constructor
        
        Mocks
        ------------
            ollama.Client is mocked and returns a mock instance
        
        Asserts
        ------------
            client.url matches the provided URL
            ollama.Client is called exactly once with correct host parameter
        """
        ## Arrange
        url = 'custom-url'

        # Client | Create a mock instance of the client
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance
        # Client | Mock the list response in client initialization to match expected result
        mock_client_instance.list.return_value = MockListResponse(models=[])

        ## Act
        # Ollama Client call replaced with mock_client
        # but mock_client_instance used when calling Client methods like client.list
        client = OllamaClient(url=url)

        ## Assert
        self.assertEqual(client.url, url)
        mock_client.assert_called_once_with(host=url)


    ## Test listing pulled models with models available
    # Utilize patch to replace all ollama.list calls in `ollama_utils.py` with mock
    @patch('ollama_utils.ollama.list')
    @patch('ollama_utils.Client')
    # Bottom patch goes with inner most argument, then next patch goes to next argument, and so forth
    def test_list_pulled_models_success(self, mock_client, mock_list):
        """
        Test successful listing of pulled models from Ollama.
        
        Verifications
        ------------
            The _list_pulled_models method correctly retrieves and returns model names
            The method properly parses the response structure
        
        Mocks
        ------------
            ollama.Client is mocked and returns a mock instance
            ollama.list returns a list of models 
                List of Ollama Model objects mocked by MockModel
        
        Asserts
        ------------
            Returned models match expected list of model names
        """
        ## Arrange
        model_names = ['model-name-0', 'model-name-1']

        # Client | Create a mock instance of the client
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance

        # list | MockListResponse mocks Ollama ListResponse object
        # Returns a list of Ollama Model objects mocked by MockModel
        mock_list.return_value = MockListResponse(models=model_names)
        # Client.list | Mock the list response in client initialization to match expected list response
        mock_client_instance.list = mock_list

        ## Act
        client = OllamaClient()
        # ollama.list call will be replaced with mock_list here
        models = client._list_pulled_models()

        ## Assert
        self.assertEqual(models, model_names)


    ## Test listing models if Ollama not connected properly
    @patch('ollama_utils.ollama.list')
    @patch('ollama_utils.Client')
    def test_list_models_error(self, mock_client, mock_list):
        """
        Test error handling when listing models fails due to Ollama connection error.
        
        Verifications
        ------------
            The _list_pulled_models method raises an exception when ollama.list fails
            Exception is propagated correctly
        
        Mocks
        ------------
            ollama.Client is mocked and returns a mock instance
            ollama.list raises the expected ConnectionError
        
        Asserts
        ------------
            ConnectionError is raised when attempting to list models
        """
        ## Arrange
        error_message = "Failed to connect to Ollama. Please check that Ollama is downloaded, running and accessible. https://ollama.com/download"

        # Client | Create a mock instance of the client
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance
        # Client.list | Mock the list response in client initialization to match expected list response
        mock_client_instance.list.return_value = MockListResponse(models=[])

        # list | Create expected response for connection error
        mock_list.side_effect = ConnectionError(error_message)

        ## Act
        client = OllamaClient()

        ## Assert
        with self.assertRaises(ConnectionError):
            client._list_pulled_models()


    ## Test pulling LM from Ollama library
    # Utilize patch to replace all ollama.pull calls in `ollama_utils.py` with mock
    @patch('ollama_utils.ollama.pull')
    @patch('ollama_utils.Client')
    def test_pull_lm_success(self, mock_client, mock_pull):
        """
        Test pulling of LM from Ollama library.
        
        Verifications
        ------------
            The _pull_lm method correctly returns a response with a success status
        
        Mocks
        ------------
            ollama.Client is mocked and returns a mock instance
            ollama.pull returns a progress reponse
                ProgressResponse mocked by MockPullResponse
        
        Asserts
        ------------
            Status of ollama.pull response matches 'success'
        """
        ## Arrange
        status_message = 'success'

        # Client | Create a mock instance of the client
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance
        # Client.list | Mock the list response in client initialization to match expected list response
        mock_client_instance.list.return_value = MockListResponse(models=[])

        # pull | Create mock instance for pull response
        mock_pull.return_value = MockPullResponse(status=status_message)

        ## Act
        client = OllamaClient()
        # ollama.pull call will be replaced with mock_pull here
        response = client._pull_lm(lm_name=model_name)

        ## Assert
        self.assertEqual(response.status, status_message)


    ## Test initializting LM when LM is available in Ollama data
    @patch('ollama_utils.ollama.pull')
    @patch('ollama_utils.ollama.list')
    @patch('ollama_utils.Client')
    def test_init_lm_existing_model(self, mock_client, mock_list, mock_pull):
        """
        Test initialization of language model when the model already exists.
        
        Verifications
        ------------
            When a model exists in the list, ollama.pull is not called
            The method correctly identifies existing models
        
        Mocks
        ------------
            ollama.Client is mocked and returns a mock instance
            ollama.list returns a list of models with the target model
                List of Ollama Model objects mocked by MockModel
            ollama.pull is mocked but not expected to be called
        
        Asserts
        ------------
            ollama.pull is not called
        """
        ## Arrange
        # Client | Create a mock instance of the client
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance

        # list | Create a mock instance for list response
        mock_list.return_value = MockListResponse(models=[model_name])
        # Client.list | Mock the list response in client initialization to match expected list response
        mock_client_instance.list = mock_list

        ## Act
        client = OllamaClient()
        client._init_lm(lm_name=model_name)

        ## Assert
        mock_pull.assert_not_called()


    ## Test initializting LM when LM isn't available in Ollama data
    @patch('ollama_utils.ollama.pull')
    @patch('ollama_utils.ollama.list')
    @patch('ollama_utils.Client')
    def test_init_lm_missing_model(self, mock_client, mock_list, mock_pull):
        """
        Test initialization of LM when the model does not exist.
        
        Verifications
        ------------
            When a model doesn't exist in the list, ollama.pull is called to pull it
            The correct model name is passed to the pull method
        
        Mocks
        ------------
            ollama.Client is mocked and returns a mock instance
            ollama.list returns a list of models without the target model
                List of Ollama Model objects mocked by MockModel
            ollama.pull is mocked to check that its called
        
        Asserts
        ------------
            ollama.pull is called exactly once with the correct model name
        """
        ## Arrange
        status_message = 'success'
        # Client | Create a mock instance of the client
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance

        # list | Create a mock instance for the list response
        mock_list.return_value = MockListResponse(models=[])
        # Client.list | Mock the list response in client initialization to match expected list response
        mock_client_instance.list = mock_list

        ## Act
        client = OllamaClient()
        client._init_lm(lm_name=model_name)

        ## Assert
        mock_pull.assert_called_once_with(model_name)


    ## Test successfully getting response
    @patch('ollama_utils.ollama.pull')
    @patch('ollama_utils.ollama.list')
    @patch('ollama_utils.Client')
    def test_get_response_success(self, mock_client, mock_list, mock_pull):
        """
        Test successful retrieval of response from Ollama chat.
        
        Verifications
        ------------
            The get_response method correctly calls the chat endpoint
            The method returns the expected content from the response
            Correct parameters are passed to the chat method
        
        Mocks
        ------------
            ollama.Client is mocked and returns a mock instance
            ollama.list is mocked but not needed for results (just need to bypass API)
            ollama.pull is mocked but not needed for results (just need to bypass API)
        
        Asserts
        ------------
            Response content matches expected text
            Chat method is called with correct model and messages parameters
        """
        ## Arrange
        response = "The sky is blue because of Rayleigh scattering."
        # Client | Create a mock instance of the client
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance

        # Client.list | Mock the list response in client initialization to match expected list response
        mock_client_instance.list.return_value = MockListResponse(models=[model_name])

        # Client.chat | Create a mock instance of chat response
        mock_chat_response = MockChatResponse(content=response)
        mock_client_instance.chat.return_value = mock_chat_response

        ## Act
        client = OllamaClient()
        # client.chat call will be replaced with mock_chat_response here
        result = client.get_response()

        ## Assert
        self.assertEqual(result, response)
        mock_client_instance.chat.assert_called_once_with(
            model=lm_name,
            messages=[{'role': 'user', 'content': message}],
            options=None
        )


    ## Test getting a None response
    @patch('ollama_utils.ollama.pull')
    @patch('ollama_utils.ollama.list')
    @patch('ollama_utils.Client')
    def test_get_response_no_content(self, mock_client, mock_list, mock_pull):
        """
        Test error handling when response content is missing.
        
        Verifications
        ------------
            The get_response method raises ValueError when response content is None
            Exception is raised appropriately when no content is returned
        
        Mocks
        ------------
            ollama.Client is mocked and returns a mock instance
            ollama.list is mocked but not needed for results (just need to bypass API)
            ollama.pull is mocked but not needed for results (just need to bypass API)
        
        Asserts
        ------------
            ValueError is raised when response has no content
        """
        ## Arrange
        # Client | Create a mock instance of the client
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance

        # Client.list | Mock the list response in client initialization to match expected list response
        mock_client_instance.list.return_value = MockListResponse(models=[model_name])

        # Client.chat | Create a mock instance of chat response
        mock_chat_response = MockChatResponse(None)
        mock_client_instance.chat.return_value = mock_chat_response

        ## Act
        client = OllamaClient()

        ## Assert
        with self.assertRaises(ValueError):
            client.get_response()



    ## Test removing fully closed think tags
    @patch('ollama_utils.Client')
    def test_remove_think_tags_success(self, mock_client):
        """
        Test successful removal of think tags from text.
        
        Verifications
        ------------
            The _remove_think_tags method correctly removes opening and closing tags
            Content between tags is preserved

        Mocks
        ------------
            ollama.Client is mocked and returns a mock instance
        
        Asserts
        ------------
            Text with tags is properly cleaned by removing the tags
        """
        ## Arrange
        example_text = "<think>Some context</think> This is actual content."
        # Client | Create a mock instance of the client
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance

        # Client.list | Mock the list response in client initialization to match expected list response
        mock_client_instance.list.return_value = MockListResponse(models=[])

        ## Act
        client = OllamaClient()
        cleaned_text = client._remove_think_tags(example_text)

        ## Assert
        self.assertEqual(cleaned_text, "This is actual content.")


    ## Test removing think tags with None input
    @patch('ollama_utils.Client')
    def test_remove_think_tags_none_input(self, mock_client):
        """
        Test error handling when input to _remove_think_tags is None.
        
        Verifications
        ------------
            The _remove_think_tags method raises ValueError when given None input
        
        Mocks
        ------------
            ollama.Client is mocked and returns a mock instance
        
        Asserts
        ------------
            ValueError is raised for None input
        """
        ## Arrange
        # Client | Create a mock instance of the client
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance

        # Client.list | Mock the list response in client initialization to match expected list response
        mock_client_instance.list.return_value = MockListResponse(models=[])

        ## Act
        client = OllamaClient()

        ## Assert
        with self.assertRaises(ValueError):
            client._remove_think_tags(None)


    ## Test removing think tags when only opening tag
    @patch('ollama_utils.Client')
    def test_remove_think_tags_only_opening(self, mock_client):
        """
        Test removal of think tags when only opening tag is present.
        
        Verifications
        ------------
            The _remove_think_tags method correctly removes all tags when only opening tag present
        
        Mocks
        ------------
            ollama.Client is mocked and returns a mock instance
        
        Asserts
        ------------
            Text with tags is properly cleaned by removing the tags
        """
        ## Arrange
        text = "<think> Some content"
        # Client | Create a mock instance of the client
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance

        # Client.list | Mock the list response in client initialization to match expected list response
        mock_client_instance.list.return_value = MockListResponse(models=[])

        ## Act
        client = OllamaClient()
        result = client._remove_think_tags(text)
        
        ## Assert
        self.assertEqual(result, "Some content")


    ## Test removing think tags when only closing tag
    @patch('ollama_utils.Client')
    def test_remove_think_tags_only_closing(self, mock_client):
        """
        Test removal of think tags when only closing tag is present.
        
        Verifications
        ------------
            The _remove_think_tags method correctly removes all tags when only closing tag present
        
        Mocks
        ------------
            ollama.Client is mocked and returns a mock instance
        
        Asserts
        ------------
            Text with tags is properly cleaned by removing the tags
        """
        ## Arrange
        text = "Some content </think>"
        # Client | Create a mock instance of the client
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance

        # Client.list | Mock the list response in client initialization to match expected list response
        mock_client_instance.list.return_value = MockListResponse(models=[])

        ## Act
        client = OllamaClient()
        result = client._remove_think_tags(text)

        ## Assert
        self.assertEqual(result, "Some content")


    ## Test removing think tags when multiple fully closed
    @patch('ollama_utils.Client')
    def test_remove_think_tags_multiple_closed(self, mock_client):
        """
        Test removal of think tags when multiple fully closed tags are present.
        
        Verifications
        ------------
            The _remove_think_tags method correctly removes all tags when multiple fully closed tags are present
        
        Mocks
        ------------
            ollama.Client is mocked and returns a mock instance
        
        Asserts
        ------------
            Text with tags is properly cleaned by removing the tags
        """
        ## Arrange
        text = "<think>First</think> and <think>Second</think>"
        # Client | Create a mock instance of the client
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance

        # Client.list | Mock the list response in client initialization to match expected list response
        mock_client_instance.list.return_value = MockListResponse(models=[])

        ## Act
        client = OllamaClient()
        result = client._remove_think_tags(text)
        
        ## Assert        
        self.assertEqual(result, "and")


    ## Test removing think tags when fully closed and one opening remaining
    @patch('ollama_utils.Client')
    def test_remove_think_tags_closed_and_opening(self, mock_client):
        """
        Test removal of think tags when fully closed tags and one opening tag are present.
        
        Verifications
        ------------
            The _remove_think_tags method correctly removes all tags when fully closed tags and one opening tag are present
        
        Mocks
        ------------
            ollama.Client is mocked and returns a mock instance
        
        Asserts
        ------------
            Text with tags is properly cleaned by removing the tags
        """
        ## Arrange
        text = "<think>First</think> and <think>"
        # Client | Create a mock instance of the client
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance

        # Client.list | Mock the list response in client initialization to match expected list response
        mock_client_instance.list.return_value = MockListResponse(models=[])

        ## Act
        client = OllamaClient()
        result = client._remove_think_tags(text)

        ## Assert
        self.assertEqual(result, "and")


    ## Test removing think tags when fully closed and one closing remaining
    @patch('ollama_utils.Client')
    def test_remove_think_tags_closed_and_closing(self, mock_client):
        """
        Test removal of think tags when fully closed tags and one closing tag are present.
        
        Verifications
        ------------
            The _remove_think_tags method correctly removes all tags when fully closed tags and one closing tag are present
        
        Mocks
        ------------
            ollama.Client is mocked and returns a mock instance
        
        Asserts
        ------------
            Text with tags is properly cleaned by removing the tags
        """
        ## Arrange
        text = "<think>First</think> and <think>"
        # Client | Create a mock instance of the client
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance

        # Client.list | Mock the list response in client initialization to match expected list response
        mock_client_instance.list.return_value = MockListResponse(models=[])

        ## Act
        client = OllamaClient()
        result = client._remove_think_tags(text)

        ## Assert
        self.assertEqual(result, "and")


if __name__ == '__main__':
    unittest.main()