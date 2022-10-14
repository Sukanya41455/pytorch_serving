import torch
import logging
import transformers
import os
import json

from ts.torch_handler.base_handler import BaseHandler
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)
logger.info(f"Transformer version {transformers.__version__}")

class ModelHandler(BaseHandler):
    def initialize(self, context):
        """Initialize function loads the model and the tokenizer

        Args:
            context (context): It is a JSON Object containing information
            pertaining to the model artifacts parameters.

        Raises:
            RuntimeError: Raises the Runtime error when the model or
            tokenizer is missing

        """
        properties = context.system_properties
        self.manifest = context.manifest
        model_dir = properties.get("model_dir")

        # get logger info with model serve
        # logger.info(f'Properties: {properties}')
        # logger.info(f'Manifest: {self.manifest}')
        # model serve logs
        # MODEL_LOG - Properties: {'model_dir': '/tmp/models/f670e3161d2e4379a7d4430d69d22719', 'gpu_id': None, 'batch_size': 1, 'server_name': 'MMS', 'server_version': '0.6.0', 'limit_max_image_pixels': True}
        # MODEL_LOG - Manifest: {'createdOn': '14/10/2022 22:10:12', 'runtime': 'python', 'model': {'modelName': 'BERTweetSentimentAnalysis', 'handler': 'handler.py', 'modelFile': 'pytorch_model.bin', 'modelVersion': '1.0'}, 'archiverVersion': '0.6.0'}
        

        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available() and properties.get("gpu_id") is not None else "cpu" 
        )
        logger.info(f"Using device {self.device}")

        # load the model
        model_file = self.manifest['model']['modelFile']
        model_path = os.path.join(model_dir, model_file)

        if os.path.isfile(model_path):
            self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f'Successfully loaded model from {model_file}')
        else:
            raise RuntimeError('Missing the model file')

        # load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        if self.tokenizer is None:
            logger.info('Successfully loaded tokenizer')
        else:
            raise RuntimeError('Missing tokenizer')

        # load mapping file
        mapping_file_path = os.path.join(model_dir, 'index_to_name.json')
        if os.path.isfile(mapping_file_path):
            with open(mapping_file_path) as f:
                self.mapping = json.load(f)
            logger.info('Successfully loaded mapping file')
        else:
            logger.warning('Mapping file is not detected')

        self.initialized = True


    def preprocess(self, requests):
        """Tokenize the input text using the suitable tokenizer and convert 
        it to tensor

        Args:
            requests: A list containing a dictionary, might be in the form
            of [{'body': json_file}] or [{'data': json_file}]
        """
        data = requests[0].get('body')
        if data is None:
            data = requests[0].get('data')

        texts = data.get('input')
        logger.info(f'Received {len(texts)} texts. Begin tokenizing')

        # tokenize the texts
        tokenized_data = self.tokenizer(texts, padding=True, return_tensors='pt')
        logger.info('Tokenization process completed')

        return tokenized_data

    def inference(self, inputs):
        """Predict class using the model

        Args:
            inputs: tensor of tokenized data
        """
        outputs = self.model(**inputs.to(self.device))
        probabilities = torch.nn.functional.sofymax(outputs.logits, dim=-1)
        predictions = torch.argmax(probabilities, axis=1)
        predictions = predictions.tolist()
        logger.info('Predictions successfully created.')

        return predictions

    def postprocess(self, outputs: list):
        """
        Convert the output to the string label provided in the label mapper (index_to_name.json)

        Args:
            outputs (list): The integer label produced by the model

        Returns:
            List: The post process function returns a list of the predicted output.
        """
        predictions = [self.mapping[str(label)] for label in outputs]
        logger.info(f'PREDICTED LABELS: {predictions}')

        return [predictions]