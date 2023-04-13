Feature: Generate vhdl templates from manifests
  Scenario: Pass through x input signal
    Given a manifest content
    And it specifies pass_through = ['x']
    And i parse the manifest content
    When i generate the template
    Then it contains the line y <- x;

  Scenario: Pass through y_address signal
    Given a manifest content
    And it specifies pass_through = ['y_address']
    And i parse the manifest content
    When i generate the template
    Then it contains the line x_address <- y_address;

  Scenario: Pass through y_address and enable
    Given a manifest content
    And it specifies pass_through = ['y_address', 'enable']
    And i parse the manifest content
    When i generate the template
    Then it contains the line x_address <- y_address; and done <- enable;

  Scenario: Parsing incorrect pass_through values
    Given a manifest content
    And it specifies pass_through = ['y']
    Then parsing the manifest throws an error

  Scenario: Parsing non_matching version requirement
    Given a manifest content with version: ==0.1
    Then parsing the manifest throws an error
