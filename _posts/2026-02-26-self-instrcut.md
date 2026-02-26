Framework for improvcing the instruction-following capabilities of pretrained LLMs by bootstrapping off their own generaitons 

1) generating task instructions 
2) deteriming if the instruction represents a classifciaiton task
3) instance generation with either an input-first or output-first approach
4) filtering low-quality data 


Given the instructions and their task type, geenerate instances for each instruction independently 

Model needs to:
- understand what the target task i s based on the instruction
- figure out what additional input fields are needed 
- generete inputs and outpus 