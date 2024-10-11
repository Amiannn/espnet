import os
import json
import torch
import random
import logging
import numpy as np

from espnet2.text.whisper_token_id_converter import OpenAIWhisperTokenIDConverter

class WhisperPrompter():
    def __init__(
            self, 
            id2context, 
            prompt_template_context,
            prompt_template_no_context,
            do_context_shuffle=False
        ):
        self.id2context                 = id2context
        self.prompt_template_context    = prompt_template_context
        self.prompt_template_no_context = prompt_template_no_context
        self.do_context_shuffle         = do_context_shuffle

        logging.info(f'do_context_shuffle: {do_context_shuffle}')

    def build_training_prompt(self, elements):
        """
            elements: {
                "idx": (int), -> index of the context element
                "confidence": (float), -> confidence of this context element
                "position": (float, float), -> position of this context element
            }
        """
        elements = elements.copy()
        element_idxs = [e['idx'] for e in elements]
        
        if len(element_idxs) == 0:
            nlp_prompt = self.prompt_template_no_context
        else:
            if elements[0]['confidence'] is not None:
                elements_confidence = [e['confidence'] for e in elements]
                # contexts = ",".join([f'{self.id2context[e]}({int(s*100)})' for e, s in zip(element_idxs, elements_confidence)])
                contexts = ",".join([f'{self.id2context[e]}' for e, s in zip(element_idxs, elements_confidence)])
            else:
                if self.do_context_shuffle:
                    random.shuffle(element_idxs)
                contexts   = ",".join([self.id2context[e] for e in element_idxs])
            nlp_prompt = f'{self.prompt_template_context}{contexts}'
        return nlp_prompt

    def build_inference_prompt(self):
        prompt_context    = f'{self.prompt_template_context}'
        prompt_no_context = f'{self.prompt_template_no_context}'
        return prompt_context, prompt_no_context