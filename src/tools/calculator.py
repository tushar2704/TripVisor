##© 2024 Tushar Aggarwal. All rights reserved.(https://tushar-aggarwal.com)
##TripVisor [Towards-GenAI] (https://github.com/Towards-GenAI)
##################################################################################################
#Importing dependencies
from langchain.tools import tool


class CalculatorTools():

  @tool("Make a calcualtion")
  def calculate(operation):
    """Useful to perform any mathematical calculations, 
    like sum, minus, multiplication, division, etc.
    The input to this tool should be a mathematical 
    expression, a couple examples are `200*7` or `5000/2*10`
    """
    return eval(operation)