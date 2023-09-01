from abc import ABC, abstractmethod

from elasticai.creator.vhdl.design.design import Design


class TestBench(Design, ABC):
    """
    We use this to generate testbenches from python modules.
    The python module, typically a layer or a component for a layer, is used to create a table of input-output pairs.
    These inputs are then used to feed the hardware design.

    The full testing workflow thus consists of the following steps:

     - instantiate the translatable software version of the unit under test(UUT)
     - instantiate the test bench design and hand it the design instance associated with the UUT
     - use ghdl to analyze, compile and run the simulation, recording the results
     - compare the results and the outputs produced by the python software module
    """

    @abstractmethod
    def input_signals(self) -> dict[str, str]:
        """e.g.
        '''python
        {'reset': 'std_logic', 'x1': 'signed(4 downto 0)'}
        '''
        """
        ...

    @abstractmethod
    def output_signals(self) -> dict[str, str]:
        ...
