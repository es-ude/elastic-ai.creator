Feature: Generate hardware implementation for an identity module

    Scenario: x_address and y_address signals have the same address width
        Given layer with 3 input features and bit width of 8
        When translating and saving hw implementation
        Then width of signal x_address and y_address are equal to 2

    Scenario: x and y signals have the same width
        Given layer with 3 input features and bit width of 8
        When translating and saving hw implementation
        Then width of signal x and y are equal to 8
