### tests/test_integration.py
## Defines integration tests for making sure methods in ./ollama_utils.py can be properly used with Ollama server

## I'm learning how to create testing suites with this file, so lots of 
## (probably obvious for a seasoned developer) comments are included 
## to clarify my confusion

## For integration tests we need to check if the methods used work with a running Ollama server
# These seem a lot easier to understand than unit tests because nothing needs to be mocked,
# we can just connect to the Ollama server and check everything 

## Note about ordering
# Ordering doesn't really seem to matter much for these tests 
# especially because the LM should be pulled in the get_response method if it doesn't exist
# However, ordering will matter in later projects when the get_response method no longer has 
# the _init_lm method (i.e. when we make agents and initialize the LM in a completely different class)
# So, I left the ordering here to remind myself of how and why to do it.

import pytest
import unittest
import requests

# Import your module
from pyfiles.ollama_utils import OllamaClient, lm_name, url, message


class TestOllamaClientIntegration(unittest.TestCase):
    """
    Integration tests for OllamaClient against a real Ollama server.

    These tests ensure that the methods in `ollama_utils.py` work correctly when communicating 
    with a running instance of the Ollama server. They verify basic functionality such as:
    
    - Client initialization
    - Model pulling and listing
    - Response generation from the LM
    
    All tests require a running Ollama server at http://localhost:11434.
    """

    ## Class resources
    # This sets up resources that are used for each test in the class
    # Typically want to include shared resources, especially if they're expensive calculations
    @classmethod
    def setUpClass(cls):
        """
        Set up class-level fixtures once for all tests.

        This setup ensures that common values are defined before any individual test runs.

        Variables
        ------------
            url: str
                URL of the Ollama server
                Defaults to 'http://localhost:11434'
            lm_name: str
                Example model name
                Defaults to 'qwen3:0.6b'
            message:
                Example message to query the model
                Defaults to 'Why is the sky blue?'
        """
        cls.test_url = url
        cls.test_lm_name = lm_name
        cls.test_message = message


    ## Set up for each test
    # This sets up whatever needs to be run before each test
    # In our case, we want to make sure the Ollama server is available,
    # and if it isn't we skip the current test and move onto the next
    def setUp(self):
        """
        Set up test fixtures before each test method.

        Verifications
        ------------
            Ollama server is accessible by attempting a request to its `/api/tags` endpoint.
            If the server is unreachable, this test will be skipped with a descriptive message.

        Raises
        ------------
            unittest.SkipTest
                When the Ollama server is not reachable at localhost:11434
        """
        error_message = "Ollama server not accessible at localhost:11434"
        # Verify Ollama server is accessible before running tests
        try:
            # Try to get a response from Ollama server /api/tags endpoint using requests
            response = requests.get(f'{self.test_url}/api/tags', timeout=5)
            # If don't get success status code, print error message
            self.assertTrue(response.status_code == 200, error_message)
        # If we get connection error, skip the test
        except requests.exceptions.ConnectionError:
            self.skipTest(error_message)


    ## Test initializing client
    # Can run this test 1st because client will need to be initialized
    # for every other test
    @pytest.mark.order(1)
    def test_client_initialization(self):
        """
        Test that OllamaClient can be initialized successfully.

        This ensures that the `OllamaClient` class can be instantiated correctly with a given URL.

        Verifications
        ------------
            OllamaClient object is not None
            URL matches the expected value
            Internal HTTP client has been created
        
        Raises
        ------------
            Exception: 
                If any verifications fail
        """
        try:
            client = OllamaClient(url=self.test_url)
            self.assertIsNotNone(client)
            self.assertEqual(client.url, self.test_url)
            self.assertIsNotNone(client.client)
        except Exception as e:
            self.fail(f"Failed to list models: {e}")


    ## Test pulling LM
    # If get_response didn't have a built-in pull mechanism, 
    # we would probably want to run this test before the rest
    # because test LM should be pulled and available 
    @pytest.mark.order(2)
    def test_pulling_model(self):
        """
        Test pulling a model from Ollama.

        Validates that the `_pull_lm()` method works correctly for downloading a specified model.

        Verifications
        ------------
            The status of the pull response is shown as successful
        
        Raises
        ------------
            Exception: 
                If any verifications fail
        """
        client = OllamaClient(url=self.test_url)
        try:
            response = client._pull_lm(lm_name=self.test_lm_name)
            self.assertEqual(response.status, 'success')
        except Exception as e:
            self.fail(f"Failed to pull model: {e}")


    ## Test listing pulled models
    def test_list_pulled_models(self):
        """
        Test listing pulled models from Ollama.

        Ensures that the `_list_pulled_models()` method returns a list of models available locally.

        Verifications
        ------------
            Return type is a list
        
        Raises
        ------------
            Exception: 
                If any verifications fail
        """
        # We already know whether client can be properly initialized with first test,
        # so we can leave it out of try - except blocks
        client = OllamaClient(url=self.test_url)
        try:
            models = client._list_pulled_models()
            # Should be a list of models names
            self.assertIsInstance(models, list)
        except Exception as e:
            self.fail(f"Failed to list models: {e}")


    ## Test initializing the LM
    def test_model_setup(self):
        """
        Test that the LM setup works.

        Ensures that the `_init_lm()` method can pull a model if it's not already present.

        Raises
        ------------
            Exception: 
                If initializing the LM fails.
        """
        client = OllamaClient(url=self.test_url)
        try:
            client._init_lm(lm_name=self.test_lm_name)
        except Exception as e:
            self.fail(f"Failed to initialize LM: {e}")


    ## Test getting a response for the test message
    def test_get_response_basic(self):
        """
        Test basic get response functionality.

        Validates that the `get_response()` method correctly generates a text response given a prompt.

        Verifications
        ------------
            Response is a string
            Length of response is greater than zero
        
        Raises
        ------------
            Exception: 
                If any verifications fail
        """
        client = OllamaClient(url=self.test_url)
        try:
            response = client.get_response(
                lm_name=self.test_lm_name, 
                message=self.test_message
            )
            self.assertIsInstance(response, str)
            self.assertGreater(len(response), 0)
        except Exception as e:
            self.fail(f"Failed to get response: {e}")


    ## Test getting response with message other than test message
    def test_get_response_with_different_message(self):
        """
        Test get response with different message.

        Ensures that the LM responds consistently when provided with varied prompts.

        Verifications
        ------------
            Response is a string
            Length of response is greater than zero
        
        Raises
        ------------
            Exception: 
                If any verifications fail
        """
        client = OllamaClient(url=self.test_url)
        try:
            response = client.get_response(
                lm_name=self.test_lm_name,
                message="Hiya!"
            )
            self.assertIsInstance(response, str)
            self.assertGreater(len(response), 0)
        except Exception as e:
            self.fail(f"Failed to get response with different message: {e}")


    ## Test getting response when user message is empty
    def test_get_response_with_empty_message(self):
        """
        Test get response with empty user message.

        Ensures that the LM responds consistently when provided with an empty prompt.

        Verifications
        ------------
            Response is a string
            Length of response is greater than zero
        
        Raises
        ------------
            Exception: 
                If any verifications fail
        """
        client = OllamaClient(url=self.test_url)
        try:
            response = client.get_response(
                lm_name=self.test_lm_name,
                message=""
            )
            self.assertIsInstance(response, str)
            self.assertGreater(len(response), 0)
        except Exception as e:
            self.fail(f"Failed to get response with empty message: {e}")


    ## Test getting responses for multiple messages to the same client
    def test_multiple_requests(self):
        """
        Test multiple sequential requests to ensure client reusability.

        Validates that the same `OllamaClient` instance can make several calls without issues.

        Verifications
        ------------
            Response for each message is a string
            Length of response for each message is greater than zero
            Length of the total number of responses is equal to three (for three messages)
        
        Raises
        ------------
            Exception: 
                If any verifications fail
        """
        client = OllamaClient(url=self.test_url)
        responses = []
        for i in range(3):
            try:
                response = client.get_response(
                    lm_name=self.test_lm_name,
                    message=f"Test message {i}"
                )
                responses.append(response)

                self.assertIsInstance(response, str)
                self.assertGreater(len(response), 0)
            except Exception as e:
                self.fail(f"Failed request {i}: {e}")

        # Want to make sure valid response was appended for each message
        self.assertEqual(len(responses), 3)


    ## Test getting response consistency
    # This could but doesn't necessarily work with non-zero temperatures, 
    # so only test for deterministic modes
    def test_response_consistency(self):
        """
        Test that responses are consistent for the same input (when temperature = 0).

        Ensures deterministic behavior by checking that identical prompts produce similar outputs.

        Verifications
        ------------
            Response for each message is a string
            Length of response for each message is greater than zero
            Two responses are equal to each other
        
        Raises
        ------------
            Exception: 
                If any verifications fail
        """
        client = OllamaClient(url=self.test_url)
        try:
            response1 = client.get_response(
                lm_name=self.test_lm_name,
                message="What is 2+2?",
                options={"temperature": 0}  # Make it deterministic
            )
            response2 = client.get_response(
                lm_name=self.test_lm_name,
                message="What is 2+2?",
                options={"temperature": 0}  # Make it deterministic
            )
            
            self.assertIsInstance(response1, str)
            self.assertGreater(len(response1), 0)
            self.assertIsInstance(response2, str)
            self.assertGreater(len(response2), 0)
            
            # Check if responses are identical
            self.assertEqual(response1, response2)
            
        except Exception as e:
            self.fail(f"Failed consistency test: {e}")


    ## Test getting responses using different LMs for the same client
    def test_client_reuse(self):
        """
        Test that the same client can be reused for different models.

        Verifies that a single `OllamaClient` instance can handle multiple requests using 
        different LMs without issues.

        Verifications
        ------------
            Response for each message and LM is a string
            Length of response for each message and LM is greater than zero
        
        Raises
        ------------
            Exception: 
                If any verifications fail
        """
        client = OllamaClient(url=self.test_url)
        try:
            # Test with default model
            response1 = client.get_response(
                lm_name=self.test_lm_name,
                message="First test"
            )            
            # Test with same client but different LM and message
            response2 = client.get_response(
                lm_name='deepseek-r1:1.5b',
                message="Second test"
            )
            self.assertIsInstance(response1, str)
            self.assertGreater(len(response1), 0)
            self.assertIsInstance(response2, str)
            self.assertGreater(len(response2), 0)
            
        except Exception as e:
            self.fail(f"Failed client reuse test: {e}")


    ## Tear down after finishing each test
    def tearDown(self):
        """
        Clean up after each test method.

        Currently performs no actions but serves as a placeholder for future cleanup logic.
        """
        # Nothing special to clean up for this test
        pass


    ## Tear down after finishing all tests
    @classmethod
    def tearDownClass(cls):
        """
        Clean up class-level fixtures.

        Currently performs no actions but serves as a placeholder for any global cleanup needed.
        """
        # Nothing special to clean up for this class
        pass


if __name__ == '__main__':
    unittest.main()