Feature: Generate template from manifest
  Scenario: Pass through x signal
    Given a manifest with pass_through = ['x']
    When generating the template
    Then it contains the line: y <- x;

  Scenario: Pass through y_address and enable
    Given a manifest with pass_through = ["y_address", "enable"]
    When generating the template
    Then it contains the lines: x_address <- y_address; and done <- enable;

  Scenario: Trying to pass through incompatible signal raises Exception
    Given a manifest with pass_through = ["y"]
    Then generating the template raises a ValueError
