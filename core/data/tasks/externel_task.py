import itertools
import random
import string
from typing import List, Literal, Iterable, Any, Dict

from core.data.tasks.task import Task


class ExternelTask(Task): # used to test externel datasets
    def __init__(
        self,
        tokenizer,
        dataset_name: Literal["ag_news", "glue-cola", "emo", "poem_sentiment", "glue-sst2"]
    ):
        super().__init__(tokenizer)
        self.dataset_name = dataset_name
        # self.input_space = input_space
        if self.dataset_name == "ag_news":
            self.output_space = ["Business", "Sci/Tech", "Sports", "World"]
        elif self.dataset_name == "glue-cola":
            self.output_space = ["acceptable", "unacceptable"]
        elif self.dataset_name == "emo":
            self.output_space = ["angry", "happy", "others", "sad"]
        elif self.dataset_name == "poem_sentiment":
            self.output_space = ["negative", "no_impact", "positive"]
        elif self.dataset_name == "glue-sst2":
            self.output_space = ["negative", "positive"]
    
    def get_data(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data
        self.input_space = [x["input"] for x in self.train_data]
        self.input_output = {d["input"]: d["output"] for d in self.train_data+self.test_data}
        self.test_output = [x["output"] for x in self.test_data]
        self.test_input = [x["input"] for x in self.test_data]
        self.train_input = [x["input"] for x in self.train_data]
        self.train_output = [x["output"] for x in self.train_data]


    # def sample_inputs(self, num_inputs: int, exclude: List[str] = ()) -> List[str]:
    #     return random.sample(set(self.input_space) - set(exclude), num_inputs)
    def compare_outputs(self, output1: Any, output2: Any) -> bool:
        output1, output2 = output1.split(), output2.split()
        if len(output1)>0 and len(output2)>0:
            output1 = output1[0]
            output2 = output2[0]
        else:
            return False
        return output1 == output2
    
    def response_outputs(self, output:Any) -> bool:
        output = output.split()
        return len(output)>0
    
    def in_space_outputs(self, output: Any) -> bool:
        output = output.split()
        if len(output)>0:
            return (output[0] in self.output_sapce)
    
    
    def sample_inputs(self, num_inputs: int, exclude: List[str] = ()) -> List[str]:
        import random
        inputs = random.sample(self.input_space, num_inputs)
        return inputs

    
    def calc_output(self, inp: str) -> str:
        return self.input_output[inp]
    
    def num_examples(self) -> int:
        return len(self.input_output)