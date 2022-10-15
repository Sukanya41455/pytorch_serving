import torch
import logging
import transformers
import os
import time
import json
from abc import ABC

from ts.torch_handler.base_handler import BaseHandler
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)
logger.info(f"Transformer version {transformers.__version__}")

class ModelHandler(BaseHandler, ABC):
    def __init__(self):
        super(ModelHandler, self).__init__()
        self.initialized = False

    def initialize(self, context):
        """Initialize function loads the model and the tokenizer

        Args:
            context (context): It is a JSON Object containing information
            pertaining to the model artifacts parameters.

        Raises:
            RuntimeError: Raises the Runtime error when the model or
            tokenizer is missing

        """
        self.manifest = context.manifest

        properties = context.system_properties
        model_dir = properties.get("model_dir")
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")

        # Read model serialize/pt file
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

        self.model.to(self.device)
        self.model.eval()

        logger.debug('Transformer model from path {0} loaded successfully'.format(model_dir))

        # Read the mapping file, index to object name
        mapping_file_path = os.path.join(model_dir, "index_to_name.json")

        if os.path.isfile(mapping_file_path):
            with open(mapping_file_path) as f:
                self.mapping = json.load(f)
        else:
            logger.warning('Missing the index_to_name.json file. Inference output will not include class name.')

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
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
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

    def handle(self, data, context):

        # It can be used for pre or post processing if needed as additional request
        # information is available in context
        
        start_time = time.time()
        
        self.context = context
        metrics = self.context.metrics
        
        data_preprocess = self.preprocess(data)
        data_inference = self.inference(data_preprocess)
        data_postprocess = self.postprocess(data_inference)
        
        
        
        stop_time = time.time()
        metrics.add_time('HandlerTime', round((stop_time - start_time) * 1000, 2), None, 'ms')
        
        return data_postprocess