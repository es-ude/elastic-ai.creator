from itertools import zip_longest
from typing import Any, Callable, Iterable
from unittest import TestCase

from torch import Tensor
from torch.nn import Module


class BrevitasModelMatcher:
    """
    self implemented Test case for comparing two brevitas models
    """

    @classmethod
    def check_equality(cls, first_model: Module, second_model: Module) -> bool:
        """
        returns if two brevitas model are equal
        Args:
            first_model (Model): brevitas model
            second_model (Model): brevitas model
        Returns:
            equality boolean
        """
        if first_model is None or second_model is None:
            raise TypeError
        if first_model == second_model:
            return True
        return cls._condition_function_applies_to_all_elements_in_iterables(
            cls._layers_are_equal, first_model, second_model
        )

    @classmethod
    def _condition_function_applies_to_all_elements_in_iterables(
        cls,
        condition_function: Callable[[Any, Any], bool],
        first_iterable: Iterable[Any],
        second_iterable: Iterable[Any],
    ) -> bool:
        """
        applies a condition function to two iterables, e.g. to the layers or to the parameters
        Args:
            condition_function (Callable[[Any, Any], bool]): function
            first_iterable (Iterable[Any]): iterable of first model, e.g. the layers
            second_iterable (Iterable[Any]): iterable of second model, e.g. the layers
        Returns:
            bool, if condition function is met
        """
        for first_item, second_item in zip_longest(first_iterable, second_iterable):
            if not condition_function(first_item, second_item):
                return False
        return True

    @classmethod
    def _layers_are_equal(
        cls, first_model_layer: Module, second_model_layer: Module
    ) -> bool:
        """
        checks if two layers of two models are equal
        Args:
            first_model_layer (Layer): layer of the first model
            second_model_layer (Layer): layer of the second model
        Returns:
            equality boolean of the layers
        """
        if type(first_model_layer) == type(second_model_layer):
            first_model_parameters = cls._get_parameters_or_empty_list(
                first_model_layer
            )
            second_model_parameters = cls._get_parameters_or_empty_list(
                second_model_layer
            )
            return cls._condition_function_applies_to_all_elements_in_iterables(
                cls._parameters_are_equal,
                first_model_parameters,
                second_model_parameters,
            )
        return False

    @classmethod
    def _parameters_are_equal(
        cls, first_model_parameter: Any, second_model_parameter: Any
    ) -> bool:
        """
        compares a parameter of two equal layers in the two models
        Args:
            first_model_parameter (Tensor): parameter of layer of first model
            second_model_parameter (Tensor): parameter of layer of first model
        Returns:
            equality boolean of the parameters
        """
        return first_model_parameter is not None and first_model_parameter.equal(
            second_model_parameter
        )

    @staticmethod
    def _has_callable_parameters(layer: Module) -> bool:
        """
        checks if the current layer has a callable parameters attribute
        Args:
            layer (Layer): brevitas layer
        Returns:
            if layer has a callable parameters attribute
        """
        return hasattr(layer, "parameters") and callable(layer.parameters)

    @classmethod
    def _get_parameters_or_empty_list(cls, layer: Module) -> Iterable:
        """
        returns parameters of an layer if it has parameters
        Args:
            layer (Layer): brevitas layer
        Returns:
            list of parameters or empty list
        """
        if cls._has_callable_parameters(layer):
            return layer.parameters()
        return []


class BrevitasModelComparisonTestCase(TestCase):
    def assertModelEqual(self, expected: Module, other: Module) -> None:
        """
        compares two brevitas model for equality
        Args:
            expected (Model): brevitas model
            other (Model): brevitas model
        """
        self.assertTrue(BrevitasModelMatcher.check_equality(expected, other))
